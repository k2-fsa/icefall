# Copyright      2021  Piotr Żelasko
# Copyright      2022-2023  Xiaomi Corporation     (Authors: Mingshuang Luo,
#                                                            Zengwei Yao)
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
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from compute_fbank_ljspeech import MyFbank, MyFbankConfig
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
    SpeechSynthesisDataset,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LJSpeechTtsDataModule:
    """
    DataModule for tts experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="TTS data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=200.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
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
            default=False,
            help="When enabled, each batch will have the "
            "field: batch['cut'] with the cuts that "
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
        logging.info("About to create train dataset")
        train = SpeechSynthesisDataset(
            return_text=False,
            return_tokens=True,
            feature_input_strategy=eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            sampling_rate = 22050
            config = MyFbankConfig(
                n_fft=1024,
                n_mels=80,
                sampling_rate=sampling_rate,
                hop_length=256,
                win_length=1024,
                f_min=0,
                f_max=8000,
            )
            train = SpeechSynthesisDataset(
                return_text=False,
                return_tokens=True,
                feature_input_strategy=OnTheFlyFeatures(MyFbank(config)),
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            sampling_rate = 22050
            config = MyFbankConfig(
                n_fft=1024,
                n_mels=80,
                sampling_rate=sampling_rate,
                hop_length=256,
                win_length=1024,
                f_min=0,
                f_max=8000,
            )
            validate = SpeechSynthesisDataset(
                return_text=False,
                return_tokens=True,
                feature_input_strategy=OnTheFlyFeatures(MyFbank(config)),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = SpeechSynthesisDataset(
                return_text=False,
                return_tokens=True,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
            )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            num_buckets=self.args.num_buckets,
            shuffle=False,
        )
        logging.info("About to create valid dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.info("About to create test dataset")
        if self.args.on_the_fly_feats:
            sampling_rate = 22050
            config = MyFbankConfig(
                n_fft=1024,
                n_mels=80,
                sampling_rate=sampling_rate,
                hop_length=256,
                win_length=1024,
                f_min=0,
                f_max=8000,
            )
            test = SpeechSynthesisDataset(
                return_text=False,
                return_tokens=True,
                feature_input_strategy=OnTheFlyFeatures(MyFbank(config)),
                return_cuts=self.args.return_cuts,
            )
        else:
            test = SpeechSynthesisDataset(
                return_text=False,
                return_tokens=True,
                feature_input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
            )
        test_sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            num_buckets=self.args.num_buckets,
            shuffle=False,
        )
        logging.info("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=test_sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ljspeech_cuts_train.jsonl.gz"
        )

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get validation cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ljspeech_cuts_valid.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "ljspeech_cuts_test.jsonl.gz"
        )
