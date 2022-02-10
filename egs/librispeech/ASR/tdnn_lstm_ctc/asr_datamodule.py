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

from lhotse import CutSet, Fbank, FbankConfig, load_manifest
from lhotse.dataset import (
    BucketingSampler,
    CutConcatenate,
    CutMix,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SingleCutSampler,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class LibriSpeechAsrDataModule:
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
            help="When enabled, use 960h LibriSpeech. "
            "Otherwise, use 100h subset.",
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
            help="The number of buckets for the BucketingSampler"
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

    def train_dataloaders(self, cuts_train: CutSet) -> DataLoader:
        logging.info("About to get Musan cuts")
        cuts_musan = load_manifest(
            self.args.manifest_dir / "cuts_musan.json.gz"
        )

        transforms = []
        if self.args.enable_musan:
            logging.info("Enable MUSAN")
            transforms.append(
                CutMix(
                    cuts=cuts_musan, prob=0.5, snr=(10, 20), preserve_id=True
                )
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

        input_transforms = []
        if self.args.enable_spec_aug:
            logging.info("Enable SpecAugment")
            logging.info(
                f"Time warp factor: {self.args.spec_aug_time_warp_factor}"
            )
            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=10,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                    max_frames_mask_fraction=0.2,
                    p=0.9
                )
            )
        else:
            logging.info("Disable SpecAugment")

        logging.info("About to create train dataset")
        train = K2SpeechRecognitionDataset(
            cut_transforms=transforms,
            input_transforms=input_transforms,
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
            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:
            logging.info("Using BucketingSampler.")
            train_sampler = BucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                bucket_method="equal_duration",
                drop_last=True,
            )
        else:
            logging.info("Using SingleCutSampler.")
            train_sampler = SingleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )
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
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                return_cuts=self.args.return_cuts,
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
            return_cuts=self.args.return_cuts,
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
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest(
            self.args.manifest_dir / "cuts_train-clean-100.json.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest(
            self.args.manifest_dir / "cuts_train-clean-360.json.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest(
            self.args.manifest_dir / "cuts_train-other-500.json.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest(self.args.manifest_dir / "cuts_dev-clean.json.gz")

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest(self.args.manifest_dir / "cuts_dev-other.json.gz")

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest(self.args.manifest_dir / "cuts_test-clean.json.gz")

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest(self.args.manifest_dir / "cuts_test-other.json.gz")


import math
import random
import numpy as np
from typing import Optional, Dict

import torch

from lhotse import CutSet

class SpecAugment(torch.nn.Module):
    """
    SpecAugment performs three augmentations:
    - time warping of the feature matrix
    - masking of ranges of features (frequency bands)
    - masking of ranges of frames (time)

    The current implementation works with batches, but processes each example separately
    in a loop rather than simultaneously to achieve different augmentation parameters for
    each example.
    """

    def __init__(
        self,
        time_warp_factor: Optional[int] = 80,
        num_feature_masks: int = 1,
        features_mask_size: int = 13,
        num_frame_masks: int = 1,
        frames_mask_size: int = 70,
        max_frames_mask_fraction: float = 0.2,
        p=0.5,
    ):
        """
        SpecAugment's constructor.

        :param time_warp_factor: parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
        :param num_feature_masks: how many feature masks should be applied. Set to ``0`` to disable.
        :param features_mask_size: the width of the feature mask (expressed in the number of masked feature bins).
            This is the ``F`` parameter from the SpecAugment paper.
        :param num_frame_masks: how many frame (temporal) masks should be applied. Set to ``0`` to disable.
        :param frames_mask_size: the width of the frame (temporal) masks (expressed in the number of masked frames).
            This is the ``T`` parameter from the SpecAugment paper.
        :param max_frames_mask_fraction: limits the size of the frame (temporal) mask to this value times the length
            of the utterance (or supervision segment).
            This is the parameter denoted by ``p`` in the SpecAugment paper.
        :param p: the probability of applying this transform.
            It is different from ``p`` in the SpecAugment paper!
        """
        super().__init__()
        assert 0 <= p <= 1
        assert num_feature_masks >= 0
        assert num_frame_masks >= 0
        assert features_mask_size > 0
        assert frames_mask_size > 0
        self.time_warp_factor = time_warp_factor
        self.num_feature_masks = num_feature_masks
        self.features_mask_size = features_mask_size
        self.num_frame_masks = num_frame_masks
        self.frames_mask_size = frames_mask_size
        self.max_frames_mask_fraction = max_frames_mask_fraction
        self.p = p

    def forward(
        self,
        features: torch.Tensor,
        supervision_segments: Optional[torch.IntTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes SpecAugment for a batch of feature matrices.

        Since the batch will usually already be padded, the user can optionally
        provide a ``supervision_segments`` tensor that will be used to apply SpecAugment
        only to selected areas of the input. The format of this input is described below.

        :param features: a batch of feature matrices with shape ``(B, T, F)``.
        :param supervision_segments: an int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features`` -- there may be either
            less or more than the batch size.
            The second dimension encoder three kinds of information:
            the sequence index of the corresponding feature matrix in `features`,
            the start frame index, and the number of frames for each segment.
        :return: an augmented tensor of shape ``(B, T, F)``.
        """
        assert len(features.shape) == 3, (
            "SpecAugment only supports batches of " "single-channel feature matrices."
        )
        features = features.clone()
        if supervision_segments is None:
            # No supervisions - apply spec augment to full feature matrices.
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(features[sequence_idx])
        else:
            # Supervisions provided - we will apply time warping only on the supervised areas.
            for sequence_idx, start_frame, num_frames in supervision_segments:
                end_frame = start_frame + num_frames
                features[sequence_idx, start_frame:end_frame] = self._forward_single(
                    features[sequence_idx, start_frame:end_frame], warp=True, mask=False
                )
            # ... and then time-mask the full feature matrices. Note that in this mode,
            # it might happen that masks are applied to different sequences/examples
            # than the time warping.
            for sequence_idx in range(features.size(0)):
                features[sequence_idx] = self._forward_single(
                    features[sequence_idx], warp=False, mask=True
                )
        return features

    def _forward_single(
        self, features: torch.Tensor, warp: bool = True, mask: bool = True
    ) -> torch.Tensor:
        """
        Apply SpecAugment to a single feature matrix of shape (T, F).
        """
        if random.random() > self.p:
            # Randomly choose whether this transform is applied
            return features
        if warp:
            if self.time_warp_factor is not None and self.time_warp_factor >= 1:
                features = time_warp(features, factor=self.time_warp_factor)
        if mask:
            from torchaudio.functional import mask_along_axis

            mean = features.mean()
            for _ in range(self.num_feature_masks):
                features = mask_along_axis(
                    features.unsqueeze(0),
                    mask_param=self.features_mask_size,
                    mask_value=mean,
                    axis=2,
                ).squeeze(0)
            _max_tot_mask_frames = self.max_frames_mask_fraction * features.size(0)
            num_frame_masks = min(self.num_frame_masks, math.ceil(_max_tot_mask_frames / self.frames_mask_size))
            max_mask_frames = min(self.frames_mask_size, _max_tot_mask_frames // num_frame_masks)
            for _ in range(num_frame_masks):
                features = mask_along_axis(
                    features.unsqueeze(0),
                    mask_param=max_mask_frames,
                    mask_value=mean,
                    axis=1,
                ).squeeze(0)
        return features

    def state_dict(self) -> Dict:
        return dict(
            time_warp_factor=self.time_warp_factor,
            num_feature_masks=self.num_feature_masks,
            features_mask_size=self.features_mask_size,
            num_frame_masks=self.num_frame_masks,
            frames_mask_size=self.frames_mask_size,
            max_frames_mask_fraction=self.max_frames_mask_fraction,
            p=self.p,
        )

    def load_state_dict(self, state_dict: Dict):
        self.time_warp_factor = state_dict.get(
            "time_warp_factor", self.time_warp_factor
        )
        self.num_feature_masks = state_dict.get(
            "num_feature_masks", self.num_feature_masks
        )
        self.features_mask_size = state_dict.get(
            "features_mask_size", self.features_mask_size
        )
        self.num_frame_masks = state_dict.get("num_frame_masks", self.num_frame_masks)
        self.frames_mask_size = state_dict.get(
            "frames_mask_size", self.frames_mask_size
        )
        self.max_frames_mask_fraction = state_dict.get(
            "max_frames_mask_fraction", self.max_frames_mask_fraction
        )
        self.p = state_dict.get("p", self.p)


def time_warp(features: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Time warping as described in the SpecAugment paper.
    Implementation based on Espresso:
    https://github.com/freewym/espresso/blob/master/espresso/tools/specaug_interpolate.py#L51

    :param features: input tensor of shape ``(T, F)``
    :param factor: time warping parameter.
    :return: a warped tensor of shape ``(T, F)``
    """
    t = features.size(0)
    if t - factor <= factor + 1:
        return features
    center = np.random.randint(factor + 1, t - factor)
    warped = np.random.randint(center - factor, center + factor + 1)
    if warped == center:
        return features
    features = features.unsqueeze(0).unsqueeze(0)
    left = torch.nn.functional.interpolate(
        features[:, :, :center, :],
        size=(warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    right = torch.nn.functional.interpolate(
        features[:, :, center:, :],
        size=(t - warped, features.size(3)),
        mode="bicubic",
        align_corners=False,
    )
    return torch.cat((left, right), dim=2).squeeze(0).squeeze(0)
