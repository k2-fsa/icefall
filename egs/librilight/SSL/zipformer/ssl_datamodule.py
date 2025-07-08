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
import glob
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import lhotse
import torch
from dataset import HubertDataset
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


class LibriLightDataModule:
    """
    DataModule for SSL experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriLight test-clean
    and test-other).

    It contains all the common data pipeline modules used in SSL
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,

    This class should be derived for specific corpora used in SSL tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="SSL data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies.",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/kmeans"),
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
            default=1000,
            help="The number of buckets for the DynamicBucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--num-cuts-for-bins-estimate",
            type=int,
            default=1000000,
            help="We will draw this many cuts to estimate the duration"
            "bins for creating similar-duration buckets. Larger number"
            "means a better estimate to the data distribution, possibly"
            "at a longer init cost.",
        )
        group.add_argument(
            "--quadratic-duration",
            type=float,
            default=None,
            help="When set, it adds an extra penalty that's quadratic"
            "in size w.r.t. a cuts duration. This helps get a more"
            "even GPU utilization across different input lengths when"
            "models have quadratic input complexity. Set between 15"
            "and 40 for transformers.",
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
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--do-normalize",
            type=str2bool,
            default=True,
            help="whether to normalize the data",
        )
        group.add_argument(
            "--random-crop",
            type=str2bool,
            default=True,
            help="always crop from the beginning if false",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = HubertDataset(
            max_sample_size=max_sample_size,
            sample_rate=sample_rate,
            label_rate=label_rate,
            random_crop=random_crop,
            pad_audio=pad_audio,
            num_classes=num_classes,
            do_normalize=do_normalize,
        )

        if self.args.bucketing_sampler:
            logging.info("Using DynamicBucketingSampler.")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                quadratic_duration=self.args.quadratic_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                num_cuts_for_bins_estimate=self.args.num_cuts_for_bins_estimate,
                drop_last=self.args.drop_last,
                world_size=world_size,
                rank=rank,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                world_size=world_size,
                rank=rank,
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

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
        max_sample_size: Optional[int] = None,
        sample_rate: float = 16000,
        label_rate: float = 50,
        random_crop: bool = True,
        pad_audio: bool = False,
        num_classes: list = [504],
        do_normalize: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = HubertDataset(
            max_sample_size=max_sample_size,
            sample_rate=sample_rate,
            label_rate=label_rate,
            random_crop=random_crop,
            pad_audio=pad_audio,
            num_classes=num_classes,
            do_normalize=do_normalize,
        )
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            quadratic_duration=self.args.quadratic_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
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

    @lru_cache()
    def all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled librilight small, medium and large cuts"
        )
        small_cuts = self.small_cuts()
        medium_cuts = self.medium_cuts()
        large_cuts = self.large_cuts()
        return CutSet.mux(
            small_cuts,
            medium_cuts,
            large_cuts,
            weights=[
                229051,  # len(small_cuts)
                2022949,  # len(medium_cuts)
                19883414,  # len(large_cuts)
            ],
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get librispeech dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    @lru_cache()
    def small_cuts(self) -> CutSet:
        logging.info("About to get librilight small cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "librilight_cuts_small.jsonl.gz"
        )

    @lru_cache()
    def medium_cuts(self) -> CutSet:
        logging.info("About to get librilight medium cuts")
        filenames = glob.glob(
            str(
                self.args.manifest_dir
                / "medium_split"
                / "librilight_cuts_medium.*.jsonl.gz"
            )
        )
        pattern = re.compile(r"librilight_cuts_medium.([0-9]+).jsonl.gz")
        idx_filenames = ((int(pattern.search(f).group(1)), f) for f in filenames)
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
        sorted_filenames = [f[1] for f in idx_filenames]
        logging.info(
            f"Loading Libri-Light medium {len(sorted_filenames)} splits in lazy mode"
        )
        return lhotse.combine(lhotse.load_manifest_lazy(p) for p in sorted_filenames)

    @lru_cache()
    def large_cuts(self) -> CutSet:
        logging.info("About to get librilight large cuts")
        filenames = glob.glob(
            str(
                self.args.manifest_dir
                / "large_split"
                / "librilight_cuts_large.*.jsonl.gz"
            )
        )
        pattern = re.compile(r"librilight_cuts_large.([0-9]+).jsonl.gz")
        idx_filenames = ((int(pattern.search(f).group(1)), f) for f in filenames)
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])
        sorted_filenames = [f[1] for f in idx_filenames]
        logging.info(
            f"Loading Libri-Light large {len(sorted_filenames)} splits in lazy mode"
        )
        return lhotse.combine(lhotse.load_manifest_lazy(p) for p in sorted_filenames)
