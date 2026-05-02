# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
    BatchIO,
)
from lhotse.utils import fix_random_seed, supervision_to_frames
from torch.utils.data import DataLoader

from icefall.utils import str2bool
from dataclasses import replace
import random

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)

class GridAsrDataModule:
    """
    DataModule for k2 VSR experiments on the GRID corpus.
    It assumes there is always one train and valid dataloader,
    and a single test dataloader for the held-out unseen speakers.

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/avhubert"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=690,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=False,
            help="When enabled, the batches will come from buckets of "
            "similar duration (saves padding frames).",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help="When enabled, use on-the-fly cut mixing and feature "
            "extraction. Will drop existing precomputed feature manifests "
            "if available.",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )
        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last batch. Used by sampler.",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=3,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
 
    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []

        if self.args.concatenate_cuts:
            logging.info(
                f"Using cut concatenation with duration factor "
                f"{self.args.duration_factor} and gap {self.args.gap}."
            )
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between
            # different utterances.
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
           
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=10,
                    features_mask_size=128,
                    num_feature_masks=2,
                    frames_mask_size=10,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=VisualFeatureInputStrategy(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        logging.info("Using SimpleCutSampler.")
        train_sampler = SimpleCutSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=self.args.shuffle,
            drop_last=self.args.drop_last,
        )
        
        logging.info("About to create train dataloader")
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        worker_init_fn = _SeedWorkers(self.args.seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        validate = K2SpeechRecognitionDataset(
            input_strategy=VisualFeatureInputStrategy(),
            cut_transforms=transforms,
            return_cuts=self.args.return_cuts,
        )
        
        valid_sampler = SimpleCutSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
        )

        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=VisualFeatureInputStrategy(),
            return_cuts=self.args.return_cuts,
        )
        sampler = SimpleCutSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=False,
            )

        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl
        
    @lru_cache()
    def train_all_cuts(self) -> CutSet:
        cuts = load_manifest_lazy(self.args.manifest_dir / "grid_cuts_train.jsonl.gz")
        cuts = cuts.map_supervisions(
            lambda s: replace(s, text=" ".join(w for w in s.text.split() if w != "sp"))
        )
        return cuts
    
    def split_train_valid(self, cuts: CutSet, valid_ratio=0.03, seed=42):
        cuts = cuts.shuffle(random.Random(seed)) 
        n = len(cuts)
        n_valid = int(n * valid_ratio)

        valid_cuts = cuts.subset(first=n_valid)
        train_cuts = cuts.subset(last=n - n_valid)

        return train_cuts, valid_cuts
    


    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("Grid: About to get test cuts")
        cuts = load_manifest_lazy(
            self.args.manifest_dir / "grid_cuts_test.jsonl.gz"
        )
        cuts = cuts.map_supervisions(
            lambda s: replace(
                s,
                text=" ".join(w for w in s.text.split() if w != "sp")
            )
        )
        return cuts

class VisualFeatureInputStrategy(BatchIO):
    def __init__(self, frame_shift: float = 0.04):
        super().__init__()
        self.frame_shift = frame_shift
        
    def __call__(self, cuts):
        feats = [torch.from_numpy(cut.load_custom("video_features")).float() for cut in cuts]
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.int32)
        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
        return feats, lengths
      
    @property
    def extractor(self):
        class DummyExtractor:
            def __init__(self, frame_shift):
                self.frame_shift = frame_shift
        return DummyExtractor(self.frame_shift)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        start_frames, nums_frames, sequence_idx = [], [], []
        for i, cut in enumerate(cuts):
            for sup in cut.supervisions:
                start, num = supervision_to_frames(
                    sup, self.frame_shift, cut.sampling_rate, max_frames=None
                )
                start_frames.append(start)
                nums_frames.append(num)
                sequence_idx.append(i)                
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_frame": torch.tensor(start_frames, dtype=torch.int32),
            "num_frames": torch.tensor(nums_frames, dtype=torch.int32),
        }
