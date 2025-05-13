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
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from lhotse import (
    CutSet,
    WhisperFbank,
    WhisperFbankConfig,
    load_manifest,
    load_manifest_lazy,
)
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SimpleCutSampler,
    SpecAugment,
)
from lhotse.dataset.input_strategies import (  # noqa F401 For AudioSamples
    AudioSamples,
    OnTheFlyFeatures,
)
from lhotse.utils import fix_random_seed
from speech_dataset import K2SpeechRecognitionDataset
from torch.utils.data import DataLoader
from utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


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
            default=Path("data/fbank"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=300.0,
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
            default=4,
            help="The number of training dataloader workers that "
            "collect the batches.",
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
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
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

        group.add_argument(
            "--huggingface-dataset-path-or-name",
            type=str,
            default="/workspace/Belle_1.4M-SLAM-Omni",
            help="The path or name of the Huggingface dataset",
        )
        group.add_argument(
            "--audio-key",
            type=str,
            default="question_audio",
            help="The key in the Huggingface dataset containing the audio data",
        )
        group.add_argument(
            "--text-key",
            type=str,
            default="answer",
            help="The key in the Huggingface dataset containing the text data",
        )
        group.add_argument(
            "--resample-to-16kHz",
            type=str2bool,
            default=True,
            help="Resample audio to 16kHz. Default: False.",
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

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(f"Time warp factor: {self.args.spec_aug_time_warp_factor}")
            # Set the value of num_frame_masks according to Lhotse's version.
            # In different Lhotse's versions, the default of num_frame_masks is
            # different.
            num_frame_masks = 10
            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]
            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2
            logging.info(f"Num frame mask: {num_frame_masks}")
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(
                WhisperFbank(WhisperFbankConfig(num_filters=80, device="cuda"))
            )
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        # if self.args.on_the_fly_feats:
        #     # NOTE: the PerturbSpeed transform should be added only if we
        #     # remove it from data prep stage.
        #     # Add on-the-fly speed perturbation; since originally it would
        #     # have increased epoch size by 3, we will apply prob 2/3 and use
        #     # 3x more epochs.
        #     # Speed perturbation probably should come first before
        #     # concatenation, but in principle the transforms order doesn't have
        #     # to be strict (e.g. could be randomized)
        #     # transforms = [PerturbSpeed(factors=[0.9, 1.1], p=2/3)] + transforms   # noqa
        #     # Drop feats to be on the safe side.
        #     train = K2SpeechRecognitionDataset(
        #         cut_transforms=transforms,
        #         input_strategy=OnTheFlyFeatures(
        #             WhisperFbank(WhisperFbankConfig(num_filters=80, device="cuda"))
        #         ),
        #         input_transforms=input_transforms,
        #         return_cuts=self.args.return_cuts,
        #     )

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
            persistent_workers=True if self.args.num_workers > 0 else False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        """
        Args:
            cuts_valid:
                CutSet for validation.
        """
        logging.info("About to create dev dataset")

        validate = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(
                WhisperFbank(WhisperFbankConfig(num_filters=80, device="cuda"))
            )
            if self.args.on_the_fly_feats
            else eval(self.args.input_strategy)(),
            return_cuts=self.args.return_cuts,
        )
        if self.args.bucketing_sampler:
            valid_sampler = DynamicBucketingSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
            )
        else:
            valid_sampler = SimpleCutSampler(
                cuts_valid,
                max_duration=self.args.max_duration,
                shuffle=False,
            )
        logging.info("About to create dev dataloader")
        valid_num_workers = 1
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=valid_num_workers,
            persistent_workers=True if valid_num_workers > 0 else False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2SpeechRecognitionDataset(
            input_strategy=OnTheFlyFeatures(
                WhisperFbank(WhisperFbankConfig(num_filters=80, device="cpu"))
            )
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

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        if self.args.on_the_fly_feats:
            pass
        else:
            return {
                "test": load_manifest_lazy(
                    self.args.manifest_dir / "cuts_belle_test.jsonl.gz"
                )
            }

    @lru_cache()
    def dev_cuts(self) -> CutSet:
        logging.info("About to get test cuts")
        if self.args.on_the_fly_feats:
            pass
        else:
            return load_manifest_lazy(
                self.args.manifest_dir / "cuts_belle_test.jsonl.gz"
            )

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        slam_omni_zh_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_belle_train.jsonl.gz"
        )
        return slam_omni_zh_cuts

    @lru_cache()
    def train_cuts_en_vocalnet(self) -> CutSet:
        logging.info("About to get train cuts")
        VoiceAssistant_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_voice_assistant_00001-00049.jsonl.gz"
        )
        ultrachat_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_ultrachat_train.jsonl.gz"
        )
        return CutSet.mux(
            VoiceAssistant_cuts,
            ultrachat_cuts,
            weights=[
                len(VoiceAssistant_cuts),
                len(ultrachat_cuts),
            ],
        )

    # valid cuts_voice_assistant.00000.jsonl.gz
    @lru_cache()
    def valid_cuts_en_vocalnet(self) -> CutSet:
        logging.info("About to get valid cuts")
        VoiceAssistant_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_voice_assistant.00000.jsonl.gz"
        )
        return VoiceAssistant_cuts

    @lru_cache()
    def test_cuts_en_vocalnet(self) -> CutSet:
        logging.info("About to get test cuts")
        VoiceAssistant_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_voice_assistant_small.00000.jsonl.gz"
        )
        return {"test": VoiceAssistant_cuts}

    def test_cuts_voicebench(
        self,
    ) -> CutSet:
        logging.info("About to get test cuts")
        VoiceAssistant_cuts = load_manifest_lazy(
            self.args.manifest_dir / "cuts_voice_assistant_small.00000.jsonl.gz"
        )
        return {"test": VoiceAssistant_cuts}

    # def train_cuts_en_vocalnet(self) -> CutSet:
    #     logging.info("About to get train cuts")
    #     VoiceAssistant_cuts = load_manifest_lazy(
    #         self.args.manifest_dir / "cuts_debug.jsonl.gz"
    #     )
    #     return VoiceAssistant_cuts

    # @lru_cache()
    # def valid_cuts_en_vocalnet(self) -> CutSet:
    #     logging.info("About to get valid cuts")
    #     VoiceAssistant_cuts = load_manifest_lazy(
    #         self.args.manifest_dir / "cuts_debug.jsonl.gz"
    #     )
    #     return VoiceAssistant_cuts

    # @lru_cache()
    # def test_cuts_en_vocalnet(self) -> CutSet:
    #     logging.info("About to get test cuts")
    #     VoiceAssistant_cuts = load_manifest_lazy(
    #         self.args.manifest_dir / "cuts_debug.jsonl.gz"
    #     )
    #     return VoiceAssistant_cuts
