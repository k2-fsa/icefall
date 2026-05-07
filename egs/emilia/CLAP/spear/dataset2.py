from typing import Callable, Dict, List, Union

import torch
from torch.utils.data.dataloader import DataLoader, default_collate

from lhotse import validate
from lhotse.cut import CutSet, MonoCut
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.dataset.collation import collate_custom_field
from lhotse.utils import compute_num_frames, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix

def str2multihot(events: List[str], n_classes=527, id_mapping=None):
    # generate multi-hot class labels
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
        at_KD: bool = False,
        sv_KD: bool = False
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
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        
        self.at_KD = at_KD
        self.sv_KD = sv_KD

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the constraints
        of max_duration and max_cuts.
        """
        validate_multi_kd(cuts)

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
        mvq_tokens, mvq_token_lens = collate_custom_field(cuts_pre_mixed, "codebook_indexes", pad_value=-100)
        
        if self.at_KD:
            at_targets = collate_custom_field(
                cuts_pre_mixed, "beats_embedding", pad_value=-100
            ) # (N,C)
        else:        
            audio_events = [c.supervisions[0].audio_event for c in cuts_pre_mixed] # the label indices are in CED format
            at_targets, _ = str2multihot(audio_events) # (N, num_events)
            
        sv_targets = None
        
        # task ids
        task_ids = [c.task_id for c in cuts_pre_mixed]
        task_ids = torch.tensor(task_ids)
        
        batch = {
            "inputs": inputs,
            "cb_indexes": mvq_tokens,
            "cb_indexes_len": mvq_token_lens,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
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