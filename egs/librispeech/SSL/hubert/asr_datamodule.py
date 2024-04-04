# Copyright      2021  Piotr Żelasko
# Copyright      2024  Xiaomi Corporation     (Author: Yifan Yang)
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
from dataset import HubertAsrDataset
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    """
    DataModule for ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,

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
            "effective batch sizes, sampling strategies.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="When enabled use 960h LibriSpeech. " "Otherwise, use 100h subset.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/wav"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
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
            "--num-workers",
            type=int,
            default=2,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--do-normalize",
            type=str2bool,
            default=True,
            help="whether to normalize the data",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        do_normalize: bool,
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
        train = HubertAsrDataset(do_normalize=do_normalize)

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
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
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet, do_normalize: bool) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = HubertAsrDataset(do_normalize=do_normalize)
        valid_sampler = DynamicBucketingSampler(
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

    def test_dataloaders(self, cuts: CutSet, do_normalize: bool) -> DataLoader:
        logging.debug("About to create test dataset")
        test = HubertAsrDataset(do_normalize=do_normalize)
        sampler = DynamicBucketingSampler(
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
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        train_clean_100_cuts = self.train_clean_100_cuts()
        train_clean_360_cuts = self.train_clean_360_cuts()
        train_other_500_cuts = self.train_other_500_cuts()
        return CutSet.mux(
            train_clean_100_cuts,
            train_clean_360_cuts,
            train_other_500_cuts,
            weights=[
                28539,  # len(train_clean_100_cuts)
                104014,  # len(train_clean_360_cuts)
                148688,  # len(train_other_500_cuts)
            ],
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )
