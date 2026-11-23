from typing import Callable, List

import torch
from lhotse import validate
from lhotse.cut import Cut, CutSet
from lhotse.dataset.input_strategies import BatchIO, PrecomputedFeatures
from lhotse.utils import ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import default_collate


class ConsistencyRegularizationSpeechRecognitionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[Cut], Cut]] = None,
        input_strategy: BatchIO = PrecomputedFeatures(),
    ):
        super().__init__()
        self.return_cuts = return_cuts
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_strategy = input_strategy

        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> dict:
        """
        Return a dict

        .. code-block::

            {
                'inputs': float tensor with shape determined by :attr:`input_strategy`:
                          - single-channel:
                            - features: (B, T, F)
                            - audio: (B, T)
                          - multi-channel: currently not supported
                'supervisions': [
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

                ],
                'aug': [
                  # it contains augmented cut info
                  {'inputs': xxx, 'supervisions': [xxx]},
                  {'inputs': xxx, 'supervisions': [xxx]},
                  {'inputs': xxx, 'supervisions': [xxx]},

                  # where xxx means it contains similar info as the non-augmented version

                  # aug[i] corresponds to self.cut_transforms[i]
                ]
            }
        """
        validate_for_asr(cuts)
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        batch = self._process(cuts)

        if self.cut_transforms:
            batch["aug"] = []

            for i, tf in enumerate(self.cut_transforms):
                transformed_cuts = cuts.map(tf)

                batch["aug"].append(self._process(transformed_cuts))

        return batch

    def _process(self, cuts: CutSet):
        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        input_tpl = self.input_strategy(cuts)
        if len(input_tpl) == 3:
            # An input strategy with fault tolerant audio reading mode.
            # "cuts" may be a subset of the original "cuts" variable,
            # that only has cuts for which we successfully read the audio.
            inputs, _, cuts = input_tpl
        else:
            inputs, _ = input_tpl

        # Get a dict of tensors that encode the positional information about supervisions
        # in the batch of feature matrices. The tensors are named "sequence_idx",
        # "start_frame/sample" and "num_frames/samples".
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)

        batch = {
            "inputs": inputs,
            "supervisions": default_collate(
                [
                    {
                        "text": supervision.text,
                    }
                    for sequence_idx, cut in enumerate(cuts)
                    for supervision in cut.supervisions
                ]
            ),
        }

        # Update the 'supervisions' field with sequence_idx and start/num frames/samples
        batch["supervisions"].update(supervision_intervals)
        if self.return_cuts:
            batch["supervisions"]["cut"] = [
                cut for cut in cuts for sup in cut.supervisions
            ]

        return batch


def validate_for_asr(cuts: CutSet) -> None:
    validate(cuts)
    tol = 2e-3  # 1ms
    for cut in cuts:
        for supervision in cut.supervisions:
            assert supervision.start >= -tol, (
                f"Supervisions starting before the cut are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )

            # Supervision start time is relative to Cut ...
            # https://lhotse.readthedocs.io/en/v0.10_e/cuts.html
            #
            # 'supervision.end' is end of supervision inside the Cut
            assert supervision.end <= cut.duration + tol, (
                f"Supervisions ending after the cut "
                f"are not supported for ASR"
                f" (sup id: {supervision.id}, cut id: {cut.id})"
            )
