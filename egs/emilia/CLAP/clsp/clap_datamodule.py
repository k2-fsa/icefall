# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2025  Yifan Yang
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
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lhotse import CutSet, combine, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    DynamicBucketingSampler,
    SimpleCutSampler,
    UnsupervisedWaveformDataset,
)
from lhotse.dataset.sampling.dynamic_bucketing import FixedBucketBatchSizeConstraint
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class DataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="CLAP data related options",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
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
            default=True,
            help="When enabled, each batch will have the "
            "field: batch['supervisions']['cut'] with the cuts that "
            "were used to construct it.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=16,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = UnsupervisedWaveformDataset()

        if self.args.bucketing_sampler:
            logging.info(
                "Using DynamicBucketingSampler with strict FixedBucketBatchSizeConstraint."
            )
            constraint = FixedBucketBatchSizeConstraint(
                max_seq_len_buckets=self.args.max_seq_len_buckets,
                batch_sizes=self.args.fixed_batch_sizes,
            )
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                constraint=constraint,
                shuffle=True,
                drop_last=True,
                duration_bins=self.args.duration_bins,
                buffer_size=self.args.num_buckets * 5000,
                sync_buckets=True,
                concurrent=False,
                world_size=world_size,
                rank=rank,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                drop_last=self.args.drop_last,
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
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=16,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(
        self,
        cuts_valid: CutSet,
        world_size: int = 1,
        rank: int = 0,
    ) -> DataLoader:
        logging.info("About to create dev dataset")
        validate = UnsupervisedWaveformDataset()
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=4,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = UnsupervisedWaveformDataset()
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
            num_workers=4,
        )
        return test_dl

    def estimate_duration_bins(
        self,
        cuts: CutSet,
        world_size: int = 1,
        rank: int = 0,
    ) -> List[float]:
        logging.info("Estimating duration bins for FixedBucketBatchSizeConstraint")

        dummy_sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            num_buckets=self.args.num_buckets,
            shuffle=True,
            drop_last=True,
            buffer_size=self.args.num_buckets * 5000,
            sync_buckets=True,
            concurrent=False,
            world_size=world_size,
            rank=rank,
        )
        duration_bins = dummy_sampler.duration_bins
        del dummy_sampler
        return duration_bins

    @lru_cache()
    def emilia_en_cuts(self) -> CutSet:
        logging.info("About to get Emilia EN tars")
        filenames = glob.glob("./download/Emilia/EN/*.tar")
        logging.info(f"Loading Emilia {len(filenames)} tars in lazy mode")
        return CutSet.from_webdataset(
            filenames,
            shuffle_shards=True,
            split_by_worker=False,
            split_by_node=False,
        )

    @lru_cache()
    def paraspeechcaps_train_base_cuts(self) -> CutSet:
        logging.info("About to get paraspeechcaps train-base shuffled cuts")
        return load_manifest_lazy(
            self.args.manifest_dir
            / "paraspeechcaps_cuts_train_base_shuf-selected.jsonl.gz"
        )

    @lru_cache()
    def paraspeechcaps_dev_cuts(self) -> CutSet:
        logging.info("About to get paraspeechcaps dev cuts")
        splits = ["voxceleb", "expresso", "ears"]
        return combine(
            load_manifest_lazy(
                self.args.manifest_dir
                / f"paraspeechcaps_cuts_dev-{s}-selected.jsonl.gz"
            )
            for s in splits
        )

    @lru_cache()
    def paraspeechcaps_test_cuts(self) -> CutSet:
        logging.info("About to get paraspeechcaps test cuts")
        splits = ["voxceleb", "expresso", "ears"]
        return combine(
            load_manifest_lazy(
                self.args.manifest_dir
                / f"paraspeechcaps_cuts_test-{s}-selected.jsonl.gz"
            )
            for s in splits
        )

    @lru_cache()
    def iemocap_cuts(self) -> CutSet:
        logging.info("About to get iemocap cuts")
        return load_manifest_lazy(self.args.manifest_dir / "iemocap_cuts_all.jsonl.gz")

    @lru_cache()
    def ravdess_cuts(self) -> CutSet:
        logging.info("About to get ravdess cuts")
        return load_manifest_lazy(self.args.manifest_dir / "ravdess_cuts_all.jsonl.gz")

    @lru_cache()
    def cremad_cuts(self) -> CutSet:
        logging.info("About to get crema-d cuts")
        return load_manifest_lazy(self.args.manifest_dir / "cremad_cuts_test.jsonl.gz")
