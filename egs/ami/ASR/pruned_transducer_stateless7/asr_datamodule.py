# Copyright      2021  Piotr Żelasko
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
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.cut import Cut
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class AmiAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).
    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction
    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description=(
                "These options are used for the preparation of "
                "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
                "effective batch sizes, sampling strategies, applied data "
                "augmentations, etc."
            ),
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help=(
                "When enabled, select noise from MUSAN and mix it "
                "with training dataset. "
            ),
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help=(
                "When enabled, utterances (cuts) will be concatenated "
                "to minimize the amount of padding."
            ),
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help=(
                "Determines the maximum duration of a concatenated cut "
                "relative to the duration of the longest cut in a batch."
            ),
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help=(
                "The amount of padding (in seconds) inserted between "
                "concatenated cuts. This padding is filled with noise when "
                "noise augmentation is used."
            ),
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=100.0,
            help=(
                "Maximum pooled recordings duration (seconds) in a "
                "single batch. You can reduce it if it causes CUDA OOM."
            ),
        )
        group.add_argument(
            "--max-cuts", type=int, default=None, help="Maximum cuts in a single batch."
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=50,
            help=(
                "The number of buckets for the BucketingSampler"
                "(you might want to increase it for larger datasets)."
            ),
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
            help=(
                "When enabled, use on-the-fly cut mixing and feature "
                "extraction. Will drop existing precomputed feature manifests "
                "if available."
            ),
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help=(
                "When enabled (=default), the examples will be "
                "shuffled for each epoch."
            ),
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=8,
            help=(
                "The number of training dataloader workers that " "collect the batches."
            ),
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
            default=80,
            help=(
                "Used only when --enable-spec-aug is True. "
                "It specifies the factor for time warping in SpecAugment. "
                "Larger values mean more warping. "
                "A value less than 1 means to disable time warp."
            ),
        )
        group.add_argument(
            "--ihm-only",
            type=str2bool,
            default=False,
            help="When enabled, only use IHM data for training.",
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
        logging.info("About to get Musan cuts")

        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

        if self.args.concatenate_cuts:
            logging.info(
                "Using cut concatenation with duration factor "
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
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=2,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=input_transforms,
            )
        else:
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_transforms=input_transforms,
            )

        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            max_cuts=self.args.max_cuts,
            shuffle=False,
            num_buckets=self.args.num_buckets,
            drop_last=True,
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

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:

        transforms = []
        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor, gap=self.args.gap
                )
            ] + transforms

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
            )
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

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else PrecomputedFeatures(),
            return_cuts=True,
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=False
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    def remove_short_cuts(self, cut: Cut) -> bool:
        """
        See: https://github.com/k2-fsa/icefall/issues/500
        Basically, the zipformer model subsamples the input using the following formula:
        num_out_frames = (num_in_frames - 7)//2
        For num_out_frames to be at least 1, num_in_frames must be at least 9.
        """
        return cut.duration >= 0.09

    @lru_cache()
    def train_cuts(self, sp: Optional[Any] = None) -> CutSet:
        logging.info("About to get AMI train cuts")

        def _remove_short_and_long_utt(c: Cut):
            if c.duration < 0.2 or c.duration > 25.0:
                return False

            # In pruned RNN-T, we require that T >= S
            # where T is the number of feature frames after subsampling
            # and S is the number of tokens in the utterance

            # In ./zipformer.py, the conv module uses the following expression
            # for subsampling
            T = ((c.num_frames - 7) // 2 + 1) // 2
            tokens = sp.encode(c.supervisions[0].text, out_type=str)
            return T >= len(tokens)

        if self.args.ihm_only:
            cuts_train = load_manifest_lazy(
                self.args.manifest_dir / "cuts_train_ihm.jsonl.gz"
            )
        else:
            cuts_train = load_manifest_lazy(
                self.args.manifest_dir / "cuts_train_all.jsonl.gz"
            )

        return cuts_train.filter(_remove_short_and_long_utt)

    @lru_cache()
    def dev_ihm_cuts(self) -> CutSet:
        logging.info("About to get AMI IHM dev cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_dev_ihm.jsonl.gz")
        return cs.filter(self.remove_short_cuts)

    @lru_cache()
    def dev_sdm_cuts(self) -> CutSet:
        logging.info("About to get AMI SDM dev cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_dev_sdm.jsonl.gz")
        return cs.filter(self.remove_short_cuts)

    @lru_cache()
    def dev_gss_cuts(self) -> CutSet:
        if not (self.args.manifest_dir / "cuts_dev_gss.jsonl.gz").exists():
            logging.info("No GSS dev cuts found")
            return None
        logging.info("About to get AMI GSS-enhanced dev cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_dev_gss.jsonl.gz")
        return cs.filter(self.remove_short_cuts)

    @lru_cache()
    def test_ihm_cuts(self) -> CutSet:
        logging.info("About to get AMI IHM test cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_test_ihm.jsonl.gz")
        return cs.filter(self.remove_short_cuts)

    @lru_cache()
    def test_sdm_cuts(self) -> CutSet:
        logging.info("About to get AMI SDM test cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_test_sdm.jsonl.gz")
        return cs.filter(self.remove_short_cuts)

    @lru_cache()
    def test_gss_cuts(self) -> CutSet:
        if not (self.args.manifest_dir / "cuts_test_gss.jsonl.gz").exists():
            logging.info("No GSS test cuts found")
            return None
        logging.info("About to get AMI GSS-enhanced test cuts")
        cs = load_manifest_lazy(self.args.manifest_dir / "cuts_test_gss.jsonl.gz")
        return cs.filter(self.remove_short_cuts)
