import math
from threading import Lock
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.utils.data.dataloader import DataLoader, default_collate
import numpy as np

from lhotse import validate
from lhotse.cut import CutSet, MonoCut
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.collation import collate_custom_field
from lhotse.utils import compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix

class CodebookCache:
    """
    Cache of 'bytes' objects with audio data.
    It is used to cache the "command" type audio inputs.

    By default it is disabled, to enable call `set_caching_enabled(True)`
    or `AudioCache.enable()`.

    The cache size is limited to max 100 elements and 500MB of audio.

    A global dict `__cache_dict` (static member variable of class AudioCache)
    is holding the codebooks as np.array.
    The key is the supervision ID, we avoid using cut.id because the cut IDs could be ruined by repeat

    Thread-safety is ensured by a threading.Lock guard.
    """

    __enabled: bool = False

    max_cache_memory: int = 500 * 1e6  # 500 MB
    max_cache_elements: int = 5000  # number audio files

    __cache_dict: Dict[str, np.array] = {}
    __lock: Lock = Lock()

    @classmethod
    def enable(cls, enabled=True):
        cls.__enabled = enabled
        if not enabled:
            cls.__clear_cache()

    @classmethod
    def enabled(cls) -> bool:
        return cls.__enabled

    @classmethod
    def try_cache(cls, key: str) -> Optional[bytes]:
        """
        Test if 'key' is in the chache. If yes return the bytes array,
        otherwise return None.
        """

        if not cls.__enabled:
            return None

        with cls.__lock:
            if key in cls.__cache_dict:
                return cls.__cache_dict[key]
            else:
                return None

    @classmethod
    def add_to_cache(cls, key: str, value: np.array):
        """
        Add the new (key,value) pair to cache.
        Possibly free some elements before adding the new pair.
        The oldest elements are removed first.
        """

        if not cls.__enabled:
            return None

        if value.itemsize * value.size > cls.max_cache_memory:
            return

        with cls.__lock:
            # limit cache elements
            while len(cls.__cache_dict) > cls.max_cache_elements:
                # remove oldest elements from cache
                # (dict pairs are sorted according to insertion order)
                cls.__cache_dict.pop(next(iter(cls.__cache_dict)))

            # limit cache memory
            while value.itemsize * value.size + CodebookCache.__cache_memory() > cls.max_cache_memory:
                # remove oldest elements from cache
                # (dict pairs are sorted according to insertion order)
                cls.__cache_dict.pop(next(iter(cls.__cache_dict)))

            # store the new (key,value) pair
            cls.__cache_dict[key] = value

    @classmethod
    def __cache_memory(cls) -> int:
        """
        Return size of CodebookCache values in bytes.
        (internal, not to be called from outside)
        """
        ans = 0
        for key, value in cls.__cache_dict.items():
            ans += value.itemsize * value.size
        return ans

    @classmethod
    def __clear_cache(cls) -> None:
        """
        Clear the cache, remove the data.
        """
        with cls.__lock:
            cls.__cache_dict.clear()

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


class MultiTaskKDDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the multi task speech and audio processing.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each item in this dataset is a dict of:

    .. code-block::

        {
            'inputs': float tensor with shape determined by :attr:`input_strategy`:
                      - single-channel:
                        - features: (B, T, F)
                        - audio: (B, T)
                      - multi-channel: currently not supported
            'supervisions': [
                {
                    'sequence_idx': Tensor[int] of shape (S,)
                    'text': List[str] of len S

                    # For feature input strategies
                    'start_frame': Tensor[int] of shape (S,)
                    'num_frames': Tensor[int] of shape (S,)

                    # For audio input strategies
                    'start_sample': Tensor[int] of shape (S,)
                    'num_samples': Tensor[int] of shape (S,)

                    # Optionally, when return_cuts=True
                    'cut': List[AnyCut] of len S
                }
            ]
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``S`` - number of supervision segments (greater or equal to B, as each Cut may have multiple supervisions)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features

    The 'sequence_idx' field is the index of the Cut used to create the example in the Dataset.
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
        target_frame_rate: int = 50,
        at_KD: bool = False,
        sv_KD: bool = False,
        enable_cache: bool = True,
    ):
        """
        IterableDataset constructor.

        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param input_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio/features.
            Examples: normalization, SpecAugment, etc.
        :param input_strategy: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk.
        :param enable_cache: Enables a cache for the codebook indexes
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        
        self.at_KD = at_KD
        self.sv_KD = sv_KD
        
        self.target_frame_rate = target_frame_rate
        self.dummy_codebook_indexes = torch.ones(1510, 16) * (-100)
        self.dummy_audio_logits = torch.ones(527) * 0.5
        
        self.enable_cache = enable_cache
        if self.enable_cache:
            CodebookCache.enable()
            assert CodebookCache.enabled()

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_duration and max_cuts.
        """
        # validate_multi_kd(cuts)

        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation that adjusts
        # the supervision boundaries.
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)

        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we succesfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        # Apply all available transforms on the inputs, i.e. either audio or features.
        # This could be feature extraction, global MVN, SpecAugment, etc.
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)
        
        # MVQ tokens
        cuts_pre_mixed = [c if isinstance(c, MonoCut) else c.tracks[0].cut for c in cuts]
        # cuts_pre_mixed = fix_start(cuts_pre_mixed)
        
        mvq_tokens, mvq_token_lens = _collate_custom_field(
            cuts_pre_mixed,
            "codebook_indexes",
            dummy=self.dummy_codebook_indexes,
            temporal_array=True,
            target_frame_rate=self.target_frame_rate,
            pad_value=-100,
        )
        
        if self.at_KD:
            # at_targets = collate_custom_field(
            #     cuts_pre_mixed, "beats_embedding", pad_value=-100
            # ) # (N,C)
            at_targets = _collate_custom_field(
                cuts_pre_mixed, "beats_embedding", dummy=self.dummy_audio_logits, temporal_array=False
            ) # (N,C)
        else:        
            audio_events = [getattr(c.supervisions[0], "audio_event", "0") for c in cuts_pre_mixed] # the label indices are in CED format
            at_targets, _ = str2multihot(audio_events) # (N, num_events)
            
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


def fix_start(cuts):
    # make the start of codebook indexes the same as the cut
    new_cuts = []
    for cut in cuts:
        if cut.has_custom("codebook_indexes"):
            cut.codebook_indexes.start = cut.start
        if cut.has_custom("firered_codebook_indexes"):
            cut.firered_codebook_indexes.start = cut.start
        new_cuts.append(cut)
    return new_cuts

def validate_multi_kd(cuts: CutSet) -> None:
    for cut in cuts:
        # assert cut.has_features, cut
        assert cut.has_custom("task_id")
        if cut.task_id == 1: 
            # speech cuts, should have codebook indexes
            assert cut.codebook_indexes.array.storage_key != "dummy_whisper_codebook_indexes_1510"
        elif cut.task_id == 2:
            # audio cuts, should have audio logits
            assert cut.beats_embedding.storage_key != "dummy_beats_embedding"

def load_codebook_indexes(c):
    info = c.codebook_indexes
    cached_cb = CodebookCache.try_cache(c.supervisions[0].id) # we use supervision ID rather than cut id because cuts.repeat() ruins the cut id
    if cached_cb is not None:
        return cached_cb
    else:
        if isinstance(info, dict):
            filename = info["path"]
            with open(filename, "rb") as f:
                cb_indexes = np.load(f)
            # return np.load(filename, mmap_mode="r")
        else:
            cb_indexes = c.load_custom("codebook_indexes")
    
        CodebookCache.add_to_cache(c.supervisions[0].id, cb_indexes)
        return cb_indexes
       
def _collate_custom_field(
    cuts: CutSet, 
    field: str,
    dummy: torch.Tensor = None,
    temporal_array: bool = True,
    target_frame_rate: int = 50,
    pad_value=None,
):
    
    # by default, we assert the frame_shift is 0.02
    if temporal_array:
        max_frames = [int(c.duration * target_frame_rate) for c in cuts]
        
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
            
        
if __name__=="__main__":
    from functools import partial
    from utils import _add_dummy_embeddings_and_taskIDs
    from lhotse import load_manifest
    
    # enable the cache
    CodebookCache.enable()
    
    dummy_codebook_indexes = torch.ones(1510, 16) * (-100)
    dummy_audio_logits = torch.ones(527) * 0.5
    
    cuts = load_manifest("data/vq_hubert_large_layer_21_normalize_1_cb_16/librispeech_cuts_dev-clean.jsonl.gz").subset(first=500).repeat(2)
    # cut_ids = [c.task_id for c in cuts]
    
    augmented_cuts = cuts.map(partial(_add_dummy_embeddings_and_taskIDs, None))
    # cuts = load_manifest("debug.jsonl.gz")
    
    # gt_mvq_tokens, gt_mvq_token_lens = collate_custom_field(augmented_cuts, "codebook_indexes", pad_value=-100)
    import time
    start = time.time()
    mvq_tokens, mvq_token_lens = _collate_custom_field(
        cuts,
        "codebook_indexes",
        dummy=dummy_codebook_indexes,
        temporal_array=True,
        pad_value=-100
    )
    print(f"Cache: {CodebookCache.enabled()}, Time elapsed: {time.time() - start}")
    # print(gt_mvq_tokens)
    
    # gt_beats_embed = collate_custom_field(augmented_cuts, "beats_embedding")
    # beats_embed = _collate_custom_field(cuts, "beats_embedding", dummy=dummy_audio_logits, temporal_array=False)
    
    # print(beats_embed)

