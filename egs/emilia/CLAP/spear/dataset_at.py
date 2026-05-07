from typing import Callable, Dict, List, Union
import random

import numpy as np

from lhotse import CutSet, load_manifest, load_manifest_lazy
from lhotse import Fbank, FbankConfig
from lhotse.dataset import CutMix
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures, OnTheFlyFeatures
from lhotse.dataset.collation import read_audio_from_cuts, collate_matrices
from lhotse.cut import MonoCut
from lhotse.utils import LOG_EPSILON, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix

import torch
import torch.utils
from torch.utils.data.dataloader import DataLoader, default_collate

def str2multihot(events: List[str], n_classes=527, id_mapping=None):
    # generate multi-hot class labels
    if not isinstance(events, list):
        events = [events]
    labels = [list(map(int, event.split(";"))) for event in events]
    batch_size = len(labels)
    out = torch.zeros(batch_size, n_classes)

    for i, label in enumerate(labels):
        if id_mapping is not None:
            label = [id_mapping[l] for l in label]
        out[i, label] = 1

    return out, labels

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        mixup_cuts: CutSet = None,
        mixup_prob: float = 0.5,
        mvq_KD: bool = False,
        at_KD: bool = False,
        sv_KD: bool = False
    ):
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.extractor = Fbank(FbankConfig(num_mel_bins=128))
        
        self.mvq_KD = mvq_KD
        self.at_KD = at_KD
        self.sv_KD = sv_KD
        
        self.mixup_cuts = mixup_cuts
        self.mixup_prob = mixup_prob
        self.dummy_codebook_indexes = torch.ones(1510, 16) * (-100)
        self.dummy_audio_logits = torch.ones(527) * 0.5

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)
        
    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)
        audios, cuts, mix_labels = self.read_and_mix_audio(cuts, p=self.mixup_prob)
        
        inputs, input_lens = compute_feature(audios, cuts, self.extractor)
        
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)
        
        # MVQ tokens
        cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
        # cuts_pre_mixed = fix_start(cuts_pre_mixed)
        
        if self.mvq_KD:
            mvq_tokens, mvq_token_lens = _collate_custom_field(
                cuts_pre_mixed,
                "codebook_indexes",
                dummy=self.dummy_codebook_indexes,
                temporal_array=True,
                pad_value=-100,
            )
        else:
            mvq_tokens = None
            mvq_token_lens = None
        
        if self.at_KD:
            # at_targets = collate_custom_field(
            #     cuts_pre_mixed, "beats_embedding", pad_value=-100
            # ) # (N,C)
            at_targets = _collate_custom_field(
                cuts_pre_mixed, "beats_embedding", dummy=self.dummy_audio_logits, temporal_array=False
            ) # (N,C)
        else:        
            at_targets = mix_labels
            
        sv_targets = None
        
        # task ids
        task_ids = [c.task_id for c in cuts_pre_mixed]
        task_ids = torch.tensor(task_ids)
        
        dummy_text = "This is dummy text."
        
        batch = {
            "inputs": inputs,
            "cb_indexes": mvq_tokens,
            "cb_indexes_len": mvq_token_lens,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text if supervision.text is not None else dummy_text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
            "task_ids": task_ids,
            "at_targets": at_targets,
            "sv_targets": sv_targets,
        }
        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        return batch
        
    def read_and_mix_audio(self, cuts: CutSet, p: float=0.5):
        audios = []
        out_cuts = []
        labels = []
        for cut in cuts:
            # mix the audio
            if random.random() < self.mixup_prob and self.mixup_cuts is not None:
                mix_cut = self.mixup_cuts.sample(n_cuts=1) 
                audio, label = _read_and_mix_audio_single(cut, mix_cut)
            else:
                audio = cut.load_audio()
                label, _ = str2multihot(cut.supervisions[0].audio_event)
            audios.append(audio)
            out_cuts.append(cut)
            labels.append(label)
            
        labels = torch.cat(labels, dim=0) # (B,num_classes)
                
        return audios, CutSet.from_cuts(out_cuts), labels
    
def _read_and_mix_audio_single(cut, mix_cut):
    mix_lambda = np.random.beta(10,10)
    audio1 = cut.load_audio()
    audio2 = mix_cut.load_audio()
    if audio1.shape[1] > audio2.shape[1]:
        diff = audio1.shape[1] - audio2.shape[1]
        padding = np.zeros((1, diff), dtype=np.float32)
        audio2 = np.concatenate((audio2, padding), axis=1)
    else:
        audio2 = audio2[:, :audio1.shape[1]]
    
    # mix the audio waveform
    mix_audio = audio1 * mix_lambda + audio2 * (1 - mix_lambda)
    
    # mix the label
    label1, _ = str2multihot(cut.supervisions[0].audio_event)
    label2, _ = str2multihot(mix_cut.supervisions[0].audio_event)
    mix_label = label1 * mix_lambda + label2 * (1 - mix_lambda)
    
    return mix_audio, mix_label

def compute_feature(audios, cuts, extractor):
    # compute features given the audios
    # cuts is only for metadata reading
    features_single = []
    for idx, (audio, cut) in enumerate(zip(audios, cuts)):
        try:
            features = extractor.extract(audio, cuts[idx].sampling_rate)
        except:
            print(
                f"Error while extracting the features for cut with ID {cut.id} -- details:\n{cut}"
            )
            raise
        features_single.append(torch.from_numpy(features))
    
    features_batch = collate_matrices(features_single, padding_value=LOG_EPSILON)
    
    feature_lens = torch.tensor(
        [f.shape[0] for f in features_single], dtype=torch.int64
    )

    out = (features_batch, feature_lens)
    return out


def load_codebook_indexes(c):
    info = c.codebook_indexes
    if isinstance(info, dict):
        filename = info["path"]
        return np.load(filename)
    else:
        return c.load_custom("codebook_indexes")
       
def _collate_custom_field(
    cuts: CutSet, 
    field: str,
    dummy: np.array = None,
    temporal_array: bool = True,
    pad_value=None,
):
    
    # by default, we assert the frame_shift is 0.02
    if temporal_array:
        max_frames = [int(c.duration * 50) for c in cuts]
        
        temporal_dim = 0
        pad_value = -100
        arrs = [
            torch.from_numpy(load_codebook_indexes(c)) if c.has_custom(field) else dummy for c in cuts # load the numpy codebook indexes
        ]
        for i, arr in enumerate(arrs):
            arrs[i] = arr[:max_frames[i],:]
        
        arr_lens = torch.tensor(
            [a.shape[temporal_dim] for a in arrs], dtype=torch.int32
        )   
        largest_arr = max(arrs, key=torch.numel)
        maxlen = largest_arr.shape[temporal_dim]
        collated_shape = (len(arrs), *largest_arr.shape)
        dtype = largest_arr.dtype
        if any(d == dtype for d in (torch.uint8, torch.int8, torch.int16, torch.int32)):
            dtype = torch.int64
        tensors = pad_value * torch.ones(collated_shape, dtype=dtype)
        for aidx, a in enumerate(arrs):
            alen = a.shape[temporal_dim]
            # Construct an index expression such as tensors[:, :alen, :, :] programmatically;
            # All indices are set to ':', besides temporal dim which is determined on pad_direction.
            
            temporal_slice = slice(0, alen)
            indices = (aidx,) + tuple(
                temporal_slice if i == temporal_dim else slice(None, None, None)
                for i in range(len(a.shape))
            )
            tensors[indices] = a

        return tensors, arr_lens
    else:
        all_arrays = [torch.from_numpy(c.load_custom(field)) if c.has_custom(field) else dummy for c in cuts]
        return torch.stack(all_arrays)
 
def test_dataset():
    mixup_cuts = load_manifest("data/fbank_as_ced_mAP50/audioset_cuts_balanced.jsonl.gz").drop_features()
    dataset = MultiTaskDataset(
        return_cuts=True,
        mixup_cuts=mixup_cuts,
        mixup_prob=0.5,
        input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=128))),
    )
    
    cuts = load_manifest("data/fbank_as_ced_mAP50/audioset_cuts_balanced.jsonl.gz").drop_features()
    cuts = cuts.subset(first=5)
    batch = dataset[cuts]
    print(batch)
    

def test_mix():
    musan_cuts = load_manifest("data/fbank/musan_cuts.jsonl.gz").drop_features()
    noise_cuts = CutSet.from_cuts([musan_cuts[0]])

    transform = CutMix(cuts=noise_cuts, p=1.0, snr=0, preserve_id=True)

    audio_cuts = load_manifest_lazy("data/fbank_as_ced_mAP50/audioset_cuts_balanced.jsonl.gz").drop_features()
    cuts = audio_cuts.subset(first=10)

    mixed_cuts = transform(cuts)
    cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
    noise_audio = noise_cuts[0].load_audio()
    
    extractor = Fbank(FbankConfig(num_mel_bins=128))

    for mixed_cut, pre_mixed_cut in zip(mixed_cuts, cuts_pre_mixed):
        
        mixed_audio = mixed_cut.load_audio()
        orig_audio = pre_mixed_cut.load_audio()
        audio_diff = mixed_audio - orig_audio
        print(mixed_audio)
        
def test_read_audio():
    audio_cuts = load_manifest_lazy("data/fbank_as_ced_mAP50/audioset_cuts_balanced.jsonl.gz").drop_features()
    cuts = audio_cuts.subset(first=10)
    
    audios, cuts = read_audio_from_cuts(cuts)
    
    print(audios)
    print(cuts)
    
        
if __name__=="__main__":
    test_dataset()