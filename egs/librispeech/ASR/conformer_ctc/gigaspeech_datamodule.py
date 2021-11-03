# Copyright (c)  2021  Johns Hopkins University (Piotr Żelasko)
# Apache 2.0
import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Union

from torch.utils.data import DataLoader

from lhotse import CutSet, Fbank, FbankConfig, load_manifest
from lhotse.dataset import (
    BucketingSampler,
    CutConcatenate,
    CutMix,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SingleCutSampler,
    SpecAugment,
)
from lhotse.dataset.dataloading import LhotseDataLoader
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from icefall.utils import str2bool
from icefall.dataset.datamodule import DataModule


def get_context_suffix(args):
    if args.context_window is None or args.context_window <= 0.0:
        ctx_suffix = ""
    else:
        ctx_suffix = f"_{args.context_direction}{args.context_window}"
    return ctx_suffix


class GigaSpeechAsrDataModule(DataModule):
    """
    DataModule for K2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean and test-other).
    It contains all the common data pipeline modules used in ASR experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction
    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args):
        self.total_train_cuts = 0
        self.consumed_cuts = 0
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        super().add_arguments(parser)
        group = parser.add_argument_group(
            title='ASR data related options',
            description='These options are used for the preparation of PyTorch DataLoaders '
                        'from Lhotse CutSet\'s -- they control the effective batch sizes, '
                        'sampling strategies, applied data augmentations, etc.'
        )
        group.add_argument(
            '--feature-dir',
            type=Path,
            default=Path('exp/data'),
            help='Path to directory with train/valid/test cuts.'
        )
        group.add_argument(
            '--max-duration',
            type=int,
            default=500.0,
            help="Maximum pooled recordings duration (seconds) in a single batch.")
        group.add_argument(
            '--bucketing-sampler',
            type=str2bool,
            default=False,
            help='When enabled, the batches will come from buckets of '
                 'similar duration (saves padding frames).')
        group.add_argument(
            '--num-buckets',
            type=int,
            default=30,
            help='The number of buckets for the BucketingSampler'
                 '(you might want to increase it for larger datasets).')
        group.add_argument(
            '--concatenate-cuts',
            type=str2bool,
            default=True,
            help='When enabled, utterances (cuts) will be concatenated '
                 'to minimize the amount of padding.')
        group.add_argument(
            '--duration-factor',
            type=float,
            default=1.0,
            help='Determines the maximum duration of a concatenated cut '
                 'relative to the duration of the longest cut in a batch.')
        group.add_argument(
            '--gap',
            type=float,
            default=1.0,
            help='The amount of padding (in seconds) inserted between concatenated cuts. '
                 'This padding is filled with noise when noise augmentation is used.')
        group.add_argument(
            '--on-the-fly-feats',
            type=str2bool,
            default=False,
            help='When enabled, use on-the-fly cut mixing and feature extraction. '
                 'Will drop existing precomputed feature manifests if available.'
        )
        group.add_argument(
            '--shuffle',
            type=str2bool,
            default=True,
            help='When enabled (=default), the examples will be shuffled for each epoch.'
            )
        group.add_argument(
            '--check-cuts',
            type=str2bool,
            default=True,
            help='When enabled (=default), we will iterate over the whole training cut set '
                 'to validate it. It should be disabled when using Apache Arrow manifests '
                 'to avoid an excessive starting time of the script with datasets>1000h.'
            )

        # GigaSpeech specific arguments
        group.add_argument(
            "--subset",
            type=str,
            default="XS",
            help="Select the GigaSpeech subset (XS|S|M|L|XL)",
        )
        group.add_argument(
            "--context-window",
            type=float,
            default=0.0,
            help="Training cut duration in seconds. "
                 "Use 0 to train on supervision segments without acoustic context, with variable cut lengths; "
                 "number larger than zero will create multi-supervisions cuts with actual acoustic context. ",
        )
        group.add_argument(
            "--context-direction",
            type=str,
            default="center",
            help="If context-window is 0, does nothing. "
                 "If it's larger than 0, determines in which direction (relative to the supervision) "
                 "to seek for extra acoustic context. Available values: (left|right|center|random).",
        )
        group.add_argument(
            '--use-context-for-test',
            type=str2bool,
            default=False,
            help='Should we read cuts with acoustic context or without it. '
                 '(note: for now, they may contain duplicated segments)'
        )
        group.add_argument(
            '--small-dev',
            type=str2bool,
            default=False,
            help='Should we use only 1000 utterances for dev (speeds up training)'
        )

    def validate_args(self):
        if self.args.subset in ['L', 'XL']:
            assert (
                self.args.shuffle == False
            ), "For GigaSpeech L/XL, you must use --shuffle 0 to avoid eagerly reading pyarrow manifests."
            assert (
                self.args.check_cuts == False
            ), "For GigaSpeech L/XL, you must use --check-cuts 0 to avoid eagerly reading pyarrow manifests."
            assert (
                self.args.bucketing_sampler == False
            ), "For GigaSpeech L/XL, you must use --bucketing-sampler 0 to avoid eagerly reading pyarrow manifests."
            assert (
                self.args.on_the_fly_feats == True
            ), "For GigaSpeech L/XL, you must use --on-the-fly-feats 1 as we do not pre-compute them by default."

    def train_dataloaders(self) -> DataLoader:
        self.validate_args()
        logging.info("About to get train cuts")
        cuts_train = self.train_cuts()
        self.total_train_cuts = len(cuts_train)
        self.consumed_cuts = 0

        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(self.args.feature_dir / 'cuts_musan.json.gz')

        logging.info("About to create train dataset")
        transforms = [CutMix(cuts=cuts_musan, prob=0.5, snr=(10, 20))]
        if self.args.concatenate_cuts:
            logging.info(f'Using cut concatenation with duration factor '
                         f'{self.args.duration_factor} and gap {self.args.gap}.')
            # Cut concatenation should be the first transform in the list,
            # so that if we e.g. mix noise in, it will fill the gaps between different utterances.
            transforms = [
                             CutConcatenate(
                                 duration_factor=self.args.duration_factor,
                                 gap=self.args.gap
                             )
                         ] + transforms

        train = K2SpeechRecognitionDataset(
            # cuts_train,
            cut_transforms=transforms,
            return_cuts=True,
            # check_inputs=self.args.check_cuts,
        )

        if self.args.on_the_fly_feats:
            # NOTE: the PerturbSpeed transform should be added only if we remove it from data prep stage.
            # # Add on-the-fly speed perturbation; since originally it would have increased epoch
            # # size by 3, we will apply prob 2/3 and use 3x more epochs.
            # # Speed perturbation probably should come first before concatenation,
            # # but in principle the transforms order doesn't have to be strict (e.g. could be randomized)
            # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2 / 3)] + transforms
            train = K2SpeechRecognitionDataset(
                cuts=cuts_train,
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), num_workers=20),
                return_cuts=True,
                # check_inputs=self.args.check_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info('Using BucketingSampler.')
            train_sampler = BucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets
            )
        else:
            logging.info('Using SingleCutSampler.')
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")
        #train_dl = DataLoader(
        #    train,
        #    sampler=train_sampler,
        #    batch_size=None,
        #    num_workers=16,
        #    persistent_workers=True,
        #)
        train_dl = LhotseDataLoader(
            train,
            sampler=train_sampler,
            num_workers=3,
            prefetch_factor=5,
        )
        return train_dl

    def inexhaustible_train_dataloaders(self):
        return self

    def valid_dataloaders(self) -> DataLoader:
        self.validate_args()
        logging.info("About to get dev cuts")
        cuts_valid = self.valid_cuts()

        transforms = [ ]
        if self.args.concatenate_cuts:
            transforms = [ CutConcatenate(
                                 duration_factor=self.args.duration_factor,
                                 gap=self.args.gap)
                          ] + transforms


        logging.info("About to create dev dataset")
        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cuts_valid,
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), num_workers=8),
                return_cuts=True,
                check_inputs=self.args.check_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                # cuts_valid,
                cut_transforms=transforms,
                return_cuts=True,
                # check_inputs=self.args.check_cuts,
            )
        valid_sampler = SingleCutSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        #valid_dl = DataLoader(
        #    validate,
        #    sampler=valid_sampler,
        #    batch_size=None,
        #    num_workers=8,
        #    persistent_workers=True,
        #)
        valid_dl = LhotseDataLoader(
            validate,
            sampler=valid_sampler,
            num_workers=2,
        )
        return valid_dl

    def test_dataloaders(self) -> Union[DataLoader, List[DataLoader]]:
        self.validate_args()
        cuts = self.test_cuts()
        is_list = isinstance(cuts, list)
        test_loaders = []
        if not is_list:
            cuts = [cuts]

        for cuts_test in cuts:
            logging.debug("About to create test dataset")
            test = K2SpeechRecognitionDataset(
                cuts_test,
                input_strategy=(
                    OnTheFlyFeatures(Fbank(FbankConfig(num_mel_bins=80)), num_workers=8)
                    if self.args.on_the_fly_feats
                    else PrecomputedFeatures()
                ),
                return_cuts=True,
                check_inputs=self.args.check_cuts,
            )
            sampler = SingleCutSampler(cuts_test, max_duration=self.args.max_duration)
            logging.debug("About to create test dataloader")
            #test_dl = DataLoader(test, batch_size=None, sampler=sampler, num_workers=1)
            test_dl = LhotseDataLoader(test, sampler=sampler, num_workers=2)
            test_loaders.append(test_dl)

        if is_list:
            return test_loaders
        else:
            return test_loaders[0]

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        # Note: for L and XL subsets, we are expecting that the training manifest is stored using pyarrow and pre-shuffled.
        cuts_path_ext = 'jsonl.gz' if self.args.subset not in ['L', 'XL'] else 'arrow'
        cuts_train = CutSet.from_file(
            self.args.feature_dir
            / f"gigaspeech_cuts_{self.args.subset}{get_context_suffix(self.args)}.{cuts_path_ext}"
        )
        return cuts_train

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        if self.args.use_context_for_test:
            path = self.args.feature_dir / f"gigaspeech_cuts_DEV{get_context_suffix(self.args)}.jsonl.gz"
        else:
            path = self.args.feature_dir / f"gigaspeech_cuts_DEV.jsonl.gz"
        logging.info(f"About to get valid cuts from {path}")
        cuts_valid = load_manifest(path)
        if self.args.small_dev:
            return cuts_valid.subset(first=1000)
        else:
            return cuts_valid

    @lru_cache()
    def test_cuts(self) -> CutSet:
        if self.args.use_context_for_test:
            path = self.args.feature_dir / f"gigaspeech_cuts_TEST{get_context_suffix(self.args)}.jsonl.gz"
        else:
            path = self.args.feature_dir / f"gigaspeech_cuts_TEST.jsonl.gz"
        logging.info(f"About to get test cuts from {path}")
        cuts_test = load_manifest(path)
        return cuts_test

    def inexhaustible_train_dataloaders(self):
      return self

    def __iter__(self):
      # work horse for inexhuastible_train_dataloaders
      while True:
        # self.total_train_cuts / self.consumed_cuts should be contained by child class
        if self.total_train_cuts == 0 and self.consumed_cuts == self.total_train_cuts:
          self.train_dl = self.train_dataloaders()
          self.consumed_cuts = 0

        for batch in self.train_dl:
          self.consumed_cuts += len(batch["supervisions"]["text"])
          yield batch
