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
import glob
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
import re
import random # to set its random seed
import numpy as np  # to set its random seed

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
)
import lhotse

# MulticopyDataset is a modified version of K2SpeechRecognitionDataset from
# lhotse.dataset, modified to, in training mode, to return a batch that has multiple
# different copies of the same data having different Musan
# augmentations and the first having none; and also include the key "num_copies"
# in the batch which would be 1 for the validation data (no Musan) and 2 for the
# different copies of the training data with musan.
from multicopy_dataset import MulticopyDataset # interface like K2SpeechRecognitionDataset

from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        random_seed = self.seed + 9999 * worker_id
        random.seed(random_seed)
        np.random.seed(random_seed)

class AsrDataModule:
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
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="""When enabled, use 960h LibriSpeech; and 10000 hour GigaSpeech if --use-giga. Otherwise, use 100h and if applicable 250h subsets.""",
        )
        group.add_argument(
            "--mini-libri",
            type=str2bool,
            default=False,
            help="True for mini librispeech",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=800.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch, including multiple copies, so if num_copies "
            "is larger the actual duration prior to making copies will be smaller."
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
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it"
            "with training dataset. ",
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )

        parser.add_argument(
            "--libri-copies",
            type=int,
            default=1,
            help="The number of copies of librispeech data used per epoch, i.e. per epoch of gigaspeech, if --use-giga=True."
            "(it is really libri_copies times 3, because of Librispeech using speed augmentation)."
        )

        parser.add_argument(
            "--use-giga",
            type=str2bool,
            default=False,
            help="If set to True, use gigaspeech in addition to librispeech.  See also --libri-copies."
        )

        parser.add_argument(
            "--use-cv",
            type=str2bool,
            default=False,
            help="If set to True, use CommonVoice in addition to librispeech.  See also --libri-copies."
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
        num_copies: int = 1,
        seed: int = 100,  # lets us specify different seed if we create data loader on different epochs.
        # note: the seed has to be the same across ranks, because the samplers need to be kept in sync
        # so we can divide up the data accurately.
        rank: int = 0,  # the torch. distributed rank, affects the seed used for

    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            logging.info("About to get Musan cuts")
            cuts_musan = load_manifest(self.args.manifest_dir / "musan_cuts.jsonl.gz")
            transforms.append(
                CutMix(cuts=cuts_musan, p=0.5, snr=(10, 20), preserve_id=True)
            )
        else:
            logging.info("Disable MUSAN")

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

        logging.info("About to create train dataset")
        train = MulticopyDataset(
            num_copies=num_copies,
            input_strategy=eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=[],
            return_cuts=self.args.return_cuts,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we
            # remove it from data prep stage.
            # Add on-the-fly speed perturbation; since originally it would
            # have increased epoch size by 3, we will apply prob 2/3 and use
            # 3x more epochs.
            # Speed perturbation probably should come first before
            # concatenation, but in principle the transforms order doesn't have
            # to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
            # Drop feats to be on the safe side.
            train = MulticopyDataset(
                num_copies=num_copies,
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                input_transforms=[],
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info(f"Using DynamicBucketingSampler, num_copies={num_copies}")
            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration / num_copies,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 2000,
                shuffle_buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
                seed=seed,
            )
        else:
            logging.info(f"Using SimpleCutSampler, num_copies={num_copies}")
            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration / num_copies,
                shuffle=self.args.shuffle,
                seed=seed,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # the data-loader workers do not have to be synchronized across the process-group,
        # we can give them rank-dependent seeds.  (There may not actually be any randomization
        # at this level in this zapformer recipe though, we do SpecAug in the main process
        # and I think the musan-related stuff happens in the sampler.
        worker_init_fn = _SeedWorkers(seed + 4321 * rank)
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
            validate = MulticopyDataset(
                num_copies=1,
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80))),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = MulticopyDataset(
                num_copies=1,
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
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
        test = MulticopyDataset(
            input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)))
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
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


class LibriSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - librispeech_cuts_dev-clean.jsonl.gz
                - librispeech_cuts_dev-other.jsonl.gz
                - librispeech_cuts_test-clean.jsonl.gz
                - librispeech_cuts_test-other.jsonl.gz
                - librispeech_cuts_train-clean-100.jsonl.gz
                - librispeech_cuts_train-clean-360.jsonl.gz
                - librispeech_cuts_train-other-500.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_clean_5_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get train-clean-5 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-clean-5.jsonl.gz"
        )

    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-clean-100.jsonl.gz"
        )

    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-clean-360.jsonl.gz"
        )

    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-other-500.jsonl.gz"
        )

    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

    def dev_clean_2_cuts(self) -> CutSet:
        logging.info("mini_librispeech: About to get dev-clean-2 cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_dev-clean-2.jsonl.gz"
        )

    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_dev-clean.jsonl.gz"
        )

    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_dev-other.jsonl.gz"
        )

    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_test-clean.jsonl.gz"
        )

    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.manifest_dir / "librispeech_cuts_test-other.jsonl.gz"
        )


class GigaSpeech:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files:

                - gigaspeech_XL_split_2000/gigaspeech_cuts_XL.*.jsonl.gz
                - gigaspeech_cuts_L.jsonl.gz
                - gigaspeech_cuts_M.jsonl.gz
                - gigaspeech_cuts_S.jsonl.gz
                - gigaspeech_cuts_XS.jsonl.gz
                - gigaspeech_cuts_DEV.jsonl.gz
                - gigaspeech_cuts_TEST.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_XL_cuts_split(self) -> CutSet:
        logging.info("About to get train-XL cuts")

        filenames = list(
            glob.glob(
                f"{self.manifest_dir}/gigaspeech_XL_split_2000/gigaspeech_cuts_XL.*.jsonl.gz"  # noqa
            )
        )

        pattern = re.compile(r"gigaspeech_cuts_XL.([0-9]+).jsonl.gz")
        idx_filenames = [(int(pattern.search(f).group(1)), f) for f in filenames]
        idx_filenames = sorted(idx_filenames, key=lambda x: x[0])

        sorted_filenames = [f[1] for f in idx_filenames]

        logging.info(f"Loading {len(sorted_filenames)} splits")

        return lhotse.combine(lhotse.load_manifest_lazy(p) for p in sorted_filenames)

    def train_XL_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_XL.jsonl.gz"
        logging.info(f"About to get train-XL cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_L_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_L.jsonl.gz"
        logging.info(f"About to get train-L cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_M_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_M.jsonl.gz"
        logging.info(f"About to get train-M cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_S_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_S.jsonl.gz"
        logging.info(f"About to get train-S cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def train_XS_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_XS.jsonl.gz"
        logging.info(f"About to get train-XS cuts from {f}")
        return CutSet.from_jsonl_lazy(f)

    def test_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_TEST.jsonl.gz"
        logging.info(f"About to get TEST cuts from {f}")
        return load_manifest_lazy(f)

    def dev_cuts(self) -> CutSet:
        f = self.manifest_dir / "gigaspeech_cuts_DEV.jsonl.gz"
        logging.info(f"About to get DEV cuts from {f}")
        return load_manifest_lazy(f)


class CommonVoice:
    def __init__(self, manifest_dir: str):
        """
        Args:
          manifest_dir:
            It is expected to contain the following files::

                - cv22-en_cuts_train.jsonl.gz
                - cv22-en_cuts_dev.jsonl.gz
                - cv22-en_cuts_test.jsonl.gz
        """
        self.manifest_dir = Path(manifest_dir)

    def train_cuts(self) -> CutSet:
        logging.info("CommonVoice: About to get train cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv22-en_cuts_train.jsonl.gz"
        )

    def dev_cuts(self) -> CutSet:
        logging.info("CommonVoice: About to get dev cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv22-en_cuts_dev.jsonl.gz"
        )

    def test_cuts(self) -> CutSet:
        logging.info("CommonVoice: About to get test cuts")
        return load_manifest_lazy(
            self.manifest_dir / "cv22-en_cuts_test.jsonl.gz"
        )
