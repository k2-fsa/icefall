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
from typing import Any, Dict, List, Optional, Union

from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class MLSEnglishHFAsrDataModule:
    """
    DataModule for MLS English ASR experiments using HuggingFace dataset.
    Handles dataset loading and provides train/valid/test dataloaders with
    on-the-fly feature extraction.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = None

    #     self._validate_args()

    # def _validate_args(self) -> None:
    #     """Validate configuration arguments."""
    #     if self.args.on_the_fly_feats is False:
    #         raise ValueError("This recipe requires on-the-fly feature extraction")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="ASR data related options",
            description="Options for data loading and processing",
        )

        # Dataset configuration
        group.add_argument(
            "--dataset-path",
            type=str,
            default="parler-tts/mls_eng",
            help="Path to HuggingFace MLS English dataset (name or local path)",
        )

        # Sampling and batching
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="Maximum batch duration in seconds",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
            help="Whether to use bucketing sampler",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="Number of buckets for DynamicBucketingSampler",
        )

        # Data augmentation
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="Whether to enable SpecAugment",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Time warp factor for SpecAugment",
        )

        # Dataloader configuration
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="Number of workers for data loading",
        )
        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=False,
            help="Whether to return cuts in batch",
        )

        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
            help="Whether to drop last incomplete batch",
        )

        return parser

    def load_dataset(self, dataset_path: Optional[str] = None) -> None:
        """Load the HuggingFace dataset."""
        dataset_path = dataset_path or self.args.dataset_path
        logging.info(f"Loading MLS English dataset from: {dataset_path}")

        try:
            from datasets import load_dataset

            self.dataset = load_dataset(dataset_path)
            logging.info("Dataset loaded successfully")
        except ImportError:
            raise ImportError("Please install datasets package: pip install datasets")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    def _create_dataset(
        self, cuts: CutSet, is_train: bool = False
    ) -> K2SpeechRecognitionDataset:
        """Create appropriate dataset with transforms."""
        transforms = []
        input_transforms = []

        if is_train and self.args.enable_spec_aug:
            input_transforms.append(self._create_spec_augment())

        return K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

    def _create_spec_augment(self) -> SpecAugment:
        """Create SpecAugment transform based on config."""
        num_frame_masks = 10
        num_frame_masks_parameter = inspect.signature(SpecAugment.__init__).parameters[
            "num_frame_masks"
        ]
        if num_frame_masks_parameter.default == 1:
            num_frame_masks = 2

        return SpecAugment(
            time_warp_factor=self.args.spec_aug_time_warp_factor,
            num_frame_masks=num_frame_masks,
            features_mask_size=27,
            num_feature_masks=2,
            frames_mask_size=100,
        )

    def _create_sampler(
        self, cuts: CutSet, shuffle: bool
    ) -> Union[DynamicBucketingSampler, SimpleCutSampler]:
        """Create appropriate sampler based on config."""
        if self.args.bucketing_sampler:
            return DynamicBucketingSampler(
                cuts,
                max_duration=self.args.max_duration,
                shuffle=shuffle,
                num_buckets=self.args.num_buckets,
                drop_last=self.args.drop_last,
            )
        return SimpleCutSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=shuffle,
        )

    def train_dataloader(
        self, sampler_state_dict: Optional[Dict[str, Any]] = None
    ) -> DataLoader:
        """Create train dataloader."""
        cuts = self.train_cuts()
        dataset = self._create_dataset(cuts, is_train=True)
        sampler = self._create_sampler(cuts, shuffle=True)

        if sampler_state_dict:
            sampler.load_state_dict(sampler_state_dict)

        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

    def valid_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        cuts = self.valid_cuts()
        return DataLoader(
            self._create_dataset(cuts),
            sampler=self._create_sampler(cuts, shuffle=False),
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        cuts = self.test_cuts()
        return DataLoader(
            self._create_dataset(cuts),
            sampler=self._create_sampler(cuts, shuffle=False),
            batch_size=None,
            num_workers=self.args.num_workers,
        )

    @lru_cache()
    def train_cuts(self) -> CutSet:
        return CutSet.from_huggingface_dataset(
            self.dataset["train"], text_key="transcript"
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        return CutSet.from_huggingface_dataset(
            self.dataset["dev"], text_key="transcript"
        )

    @lru_cache()
    def test_cuts(self) -> CutSet:
        return CutSet.from_huggingface_dataset(
            self.dataset["test"], text_key="transcript"
        )
