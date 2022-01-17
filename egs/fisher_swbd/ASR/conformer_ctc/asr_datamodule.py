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
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    BucketingSampler,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PerturbSpeed,
    PrecomputedFeatures,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class Resample16kHz:
    def __call__(self, cuts: CutSet) -> CutSet:
        return cuts.resample(16000).with_recording_path_prefix('download')


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
            "--num-buckets",
            type=int,
            default=30,
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=True,
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
            "--num-workers",
            type=int,
            default=8,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )

    def train_dataloaders(self, cuts_train: CutSet) -> DataLoader:
        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(
            self.args.manifest_dir / "musan_cuts.jsonl.gz"
        )

        input_strategy = PrecomputedFeatures()
        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(
                Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000)),
            )

        train = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            cut_transforms=[
                PerturbSpeed(factors=[0.9, 1.1], p=2 / 3, preserve_id=True),
                Resample16kHz(),
                CutMix(
                    cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True
                ),
            ],
            input_transforms=[
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=2,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            ],
            return_cuts=True,
        )

        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=self.args.shuffle,
            num_buckets=self.args.num_buckets,
            drop_last=True,
        )
        train_sampler.filter(lambda cut: 1.0 <= cut.duration <= 15.0)

        logging.info("About to create train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:

        logging.info("About to create dev dataset")
        input_strategy = PrecomputedFeatures()
        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(
                Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000)),
            )

        validate = K2SpeechRecognitionDataset(
            return_cuts=True,
            input_strategy=input_strategy,
            cut_transforms=[
                Resample16kHz(),
            ],
        )

        valid_sampler = BucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")

        input_strategy = PrecomputedFeatures()
        if self.args.on_the_fly_feats:
            input_strategy = OnTheFlyFeatures(
                Fbank(FbankConfig(num_mel_bins=80, sampling_rate=16000)),
            )

        test = K2SpeechRecognitionDataset(
            return_cuts=True,
            input_strategy=input_strategy,
            cut_transforms=[
                Resample16kHz(),
            ],
        )
        sampler = BucketingSampler(
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

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train Fisher + SWBD cuts")
        return load_manifest_lazy(
            self.args.manifest_dir
            / "train_utterances_fisher-swbd_cuts.jsonl.gz"
        )

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get dev Fisher + SWBD cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "dev_utterances_fisher-swbd_cuts.jsonl.gz"
        )

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        raise NotImplemented


def test():
    parser = argparse.ArgumentParser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    adm = AsrDataModule(args)

    cuts = adm.train_cuts()
    dl = adm.train_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break

    cuts = adm.dev_cuts()
    dl = adm.valid_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break


if __name__ == '__main__':
    test()
