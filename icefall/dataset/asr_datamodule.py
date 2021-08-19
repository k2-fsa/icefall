import argparse
import logging
from pathlib import Path
from typing import List, Union

from lhotse import Fbank, FbankConfig, load_manifest
from lhotse.dataset import (
    BucketingSampler,
    CutConcatenate,
    CutMix,
    K2SpeechRecognitionDataset,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from icefall.dataset.datamodule import DataModule
from icefall.utils import str2bool


class AsrDataModule(DataModule):
    """
    DataModule for K2 ASR experiments.
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

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--feature-dir",
            type=Path,
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=500.0,
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
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=True,
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

    def train_dataloaders(self) -> DataLoader:
        logging.info("About to get train cuts")
        cuts_train = self.train_cuts()

        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(self.args.feature_dir / "cuts_musan.json.gz")

        logging.info("About to create train dataset")
        transforms = [CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20))]
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

        input_transforms = [
            SpecAugment(
                num_frame_masks=2,
                features_mask_size=27,
                num_feature_masks=2,
                frames_mask_size=100,
            )
        ]

        train = K2SpeechRecognitionDataset(
            return_cuts=True,
            cut_transforms=transforms,
            input_transforms=input_transforms,
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
            cuts_train = cuts_train.drop_features()
            train = K2SpeechRecognitionDataset(
                cuts=cuts_train,
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
                input_transforms=input_transforms,
            )

        if self.args.bucketing_sampler:
            logging.info("Using BucketingSampler.")
            train_sampler = BucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=True,
                num_buckets=self.args.num_buckets,
                drop_last=True,
                bucket_method="equal_duration",
            )
        else:
            logging.info("Using SingleCutSampler.")
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=True,
            )
        logging.info("About to create train dataloader")
        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=False,
        )
        return train_dl

    def valid_dataloaders(self) -> DataLoader:
        logging.info("About to get dev cuts")
        cuts_valid = self.valid_cuts()

        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            cuts_valid = cuts_valid.drop_features()
            validate = K2SpeechRecognitionDataset(
                cuts_valid.drop_features(),
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
            )
        else:
            validate = K2SpeechRecognitionDataset(cuts_valid)
        valid_sampler = SingleCutSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
        )
        return valid_dl

    def test_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        cuts = self.test_cuts()
        is_list = isinstance(cuts, list)
        test_loaders = []
        if not is_list:
            cuts = [cuts]

        for cuts_test in cuts:
            logging.debug("About to create test dataset")
            test = K2SpeechRecognitionDataset(
                cuts_test,
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
            )
            sampler = SingleCutSampler(
                cuts_test, max_duration=self.args.max_duration
            )
            logging.debug("About to create test dataloader")
            test_dl = DataLoader(
                test, batch_size=None, sampler=sampler, num_workers=1
            )
            test_loaders.append(test_dl)

        if is_list:
            return test_loaders
        else:
            return test_loaders[0]
