# Copyright      2021  Piotr Żelasko
# Copyright      2022  Xiaomi Corporation     (Author: Mingshuang Luo)
# Copyright      2026  (author: Ibrahim Almajai)
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
import random
from dataclasses import replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from lhotse import CutSet, load_manifest, load_manifest_lazy
from lhotse.dataset import (  # noqa F401 for PrecomputedFeatures
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    PrecomputedFeatures,
    SimpleCutSampler,
)
from lhotse.dataset.input_strategies import BatchIO
from lhotse.utils import fix_random_seed, supervision_to_frames
from torch.utils.data import DataLoader

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class GridAsrDataModule:
    """
    DataModule for k2 VSR experiments on the GRID corpus.
    It assumes there is always one train and valid dataloader,
    and a single test dataloader for the held-out unseen speakers.

    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,

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
            default=Path("data/avhubert_mf_l8"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=384,
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
            default=0.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--roi-mode",
            choices=["centroid", "meanface"],
            default="meanface",
            help="""Which cached mouth ROIs the on-the-fly path reads.
            'centroid': <clip>.mouth_frames.npz (already transform()ed floats).
            'meanface': <clip>.mfroi.npz (raw uint8 mean-face-aligned ROIs, so
            the AV-HuBERT normalisation is applied here). Must match the
            geometry the features/checkpoint correspond to.""",
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
            "--enable-aug",
            type=str2bool,
            default=False,
            help="When enabled, augment the training mouth-ROI frames with "
            "AVHubertAugment (geometric, photometric, temporal and noise "
            "transforms) before AV-HuBERT feature extraction. Needs "
            "--on-the-fly-feats true.",
        )
        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
            help="AudioSamples or PrecomputedFeatures",
        )
        group.add_argument(
            "--avhubert-code-dir",
            type=Path,
            default="av_hubert",
            help="Path to the AV-HuBERT source directory (added temporarily to sys.path).",
        )
        group.add_argument(
            "--avhubert-ckpt",
            type=Path,
            default="download/avhubert-ckpts/base_vox_iter5.pt",
            help="Path to the AV-HuBERT pretrained checkpoint (.pt file).",
        )
        group.add_argument(
            "--layer",
            type=int,
            default=8,
            help="Number of encoder layers to keep (0-indexed upper bound). "
            "Default: %(default)s",
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

        if self.args.enable_aug and not self.args.on_the_fly_feats:
            raise ValueError(
                "--enable-aug augments only on the on-the-fly path "
                "(--on-the-fly-feats true); with precomputed features it "
                "would silently train without augmentation"
            )
        input_transforms = []
        if self.args.enable_aug:
            logging.info("Enable mouth-ROI augmentation (AVHubertAugment)")
        else:
            logging.info("Disable mouth-ROI augmentation")

        logging.info("About to create train dataset")
        if self.args.on_the_fly_feats:
            # On-the-fly ROI -> feature extraction. Optionally augment the raw
            # mouth-ROI frames before the AV-HuBERT feature extractor runs.
            augment = None
            if self.args.enable_aug:
                augment = AVHubertAugment()

            input_strategy = VisualFeatureInputStrategy(
                mode=VisualInputMode.ON_THE_FLY,
                augment=augment,
                roi_mode=self.args.roi_mode,
            )
        else:
            input_strategy = VisualFeatureInputStrategy()

        train = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            cut_transforms=transforms,
            input_transforms=input_transforms,
            return_cuts=self.args.return_cuts,
        )

        logging.info("Using SimpleCutSampler.")
        train_sampler = SimpleCutSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=self.args.shuffle,
            drop_last=self.args.drop_last,
        )

        logging.info("About to create train dataloader")
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        worker_init_fn = _SeedWorkers(self.args.seed)

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
        validate = K2SpeechRecognitionDataset(
            input_strategy=VisualFeatureInputStrategy(),
            cut_transforms=transforms,
            return_cuts=self.args.return_cuts,
        )

        valid_sampler = SimpleCutSampler(
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

    def test_dataloaders(self, cuts: CutSet, frames: bool = False) -> DataLoader:
        """frames=True serves ROI frames (no augmentation) instead of
        precomputed features, for a fine-tuned front-end."""
        logging.debug("About to create test dataset")
        if frames:
            input_strategy = VisualFeatureInputStrategy(
                mode=VisualInputMode.ON_THE_FLY, roi_mode=self.args.roi_mode
            )
        else:
            input_strategy = VisualFeatureInputStrategy()
        test = K2SpeechRecognitionDataset(
            input_strategy=input_strategy,
            return_cuts=self.args.return_cuts,
        )
        sampler = SimpleCutSampler(
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
    def train_all_cuts(self) -> CutSet:
        cuts = load_manifest_lazy(self.args.manifest_dir / "grid_cuts_train.jsonl.gz")
        cuts = cuts.map_supervisions(
            lambda s: replace(s, text=" ".join(w for w in s.text.split() if w != "sp"))
        )
        return cuts

    def split_train_valid(self, cuts: CutSet, valid_ratio=0.03, seed=42):
        cuts = cuts.shuffle(random.Random(seed))
        n = len(cuts)
        n_valid = int(n * valid_ratio)

        valid_cuts = cuts.subset(first=n_valid)
        train_cuts = cuts.subset(last=n - n_valid)

        return train_cuts, valid_cuts

    @lru_cache()
    def test_cuts(self) -> CutSet:
        logging.info("Grid: About to get test cuts")
        cuts = load_manifest_lazy(self.args.manifest_dir / "grid_cuts_test.jsonl.gz")
        cuts = cuts.map_supervisions(
            lambda s: replace(s, text=" ".join(w for w in s.text.split() if w != "sp"))
        )
        return cuts


class AVHubertAugment:
    """
    Augmentation pipeline for greyscale lip-ROI sequences (T, H, W) from the GRID corpus.

    Design principles:
    - All spatial transforms use a single sample per sequence (temporal consistency)
    - Augmentations are ordered: geometric -> photometric -> temporal -> noise
    - Defaults are set for 88x88 mouth ROIs
    - Each augmentation has its own probability, so any one can be ablated

    Args:
      speed_perturb_prob: probability of a speed change (0.85x to 1.15x) made
        by dropping or duplicating frames, keeping the length.
      crop_size: side of the random crop, resized back to output_size.
      flip_prob: probability of a horizontal flip.
      rotation_prob, rotation_degrees: probability of a rotation, and its
        maximum angle either way.
      time_mask_prob, time_mask_max_frames, time_mask_value: probability of
        masking a run of up to time_mask_max_frames frames, and the fill value.
      spatial_mask_prob, spatial_mask_ratio, mask_fill_value: probability of
        masking a random fraction spatial_mask_ratio of pixels (the same
        pixels in every frame), and the fill value.
      noise_prob, noise_std: probability and standard deviation of additive
        Gaussian noise.
      blur_prob, blur_kernel_size: probability and kernel size (made odd) of
        a Gaussian blur.
      output_size: side of the returned frames.
      mustache_prob: probability of a moustache-shaped occlusion over the
        upper lip; needs landmarks.
    """

    def __init__(
        self,
        speed_perturb_prob=1.0,
        crop_size: int = 80,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.3,
        rotation_degrees: float = 3.0,
        time_mask_prob: float = 0.3,
        time_mask_max_frames: int = 3,
        time_mask_value: float = 0.0,
        spatial_mask_prob: float = 0.1,
        spatial_mask_ratio: float = 0.05,
        mask_fill_value: float = 0.0,
        noise_prob: float = 0.1,
        noise_std: float = 0.12,
        blur_prob: float = 0.05,
        blur_kernel_size: int = 3,
        output_size: int = 88,
        mustache_prob: float = 0.0,
    ):
        assert crop_size < output_size, "crop_size must be < output_size"
        self.speed_perturb_prob = speed_perturb_prob
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees

        self.time_mask_prob = time_mask_prob
        self.time_mask_max_frames = time_mask_max_frames
        self.time_mask_value = time_mask_value

        self.spatial_mask_prob = spatial_mask_prob
        self.spatial_mask_ratio = spatial_mask_ratio
        self.mask_fill_value = mask_fill_value

        self.noise_prob = noise_prob
        self.noise_std = noise_std

        self.blur_prob = blur_prob
        # torchvision requires odd kernel size
        self.blur_kernel_size = (
            blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        )

        self.output_size = output_size
        self.mustache_prob = mustache_prob

    def _rotate(self, frames: torch.Tensor) -> torch.Tensor:
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        rotated = TF.rotate(
            frames.unsqueeze(1),
            angle=angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(1)
        return rotated

    def _random_crop_resize(self, frames: torch.Tensor) -> torch.Tensor:
        """Consistent crop across all frames, then resize back."""
        T, H, W = frames.shape
        top = torch.randint(0, H - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, W - self.crop_size + 1, (1,)).item()
        frames = frames[:, top : top + self.crop_size, left : left + self.crop_size]
        frames = TF.resize(
            frames.unsqueeze(1),
            [self.output_size, self.output_size],
            antialias=True,
        ).squeeze(1)
        return frames

    def _horizontal_flip(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.flip(frames, dims=[2])

    def _gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply identical blur to every frame."""
        k = self.blur_kernel_size
        sigma = random.uniform(0.3, 0.6)
        # torchvision transforms need a channel axis: (T, 1, H, W)
        blurred = TF.gaussian_blur(
            frames.unsqueeze(1), kernel_size=[k, k], sigma=sigma
        ).squeeze(1)
        return blurred

    def _time_masking(self, frames: torch.Tensor) -> torch.Tensor:
        """
        SpecAugment-style masking along the time axis.
        Masks a contiguous block of frames with a constant fill value.
        Applied at most once per sequence.
        """
        T = frames.shape[0]
        mask_len = torch.randint(1, self.time_mask_max_frames + 1, (1,)).item()
        mask_len = min(mask_len, T)
        start = torch.randint(0, T - mask_len + 1, (1,)).item()
        frames = frames.clone()
        frames[start : start + mask_len] = self.time_mask_value
        return frames

    def _spatial_mask(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Randomly zero a fixed fraction of spatial positions,
        using the same mask for all frames in the sequence.
        Encourages the model not to over-rely on any single pixel region.
        """
        T, H, W = frames.shape
        n_pixels = H * W
        n_masked = int(n_pixels * self.spatial_mask_ratio)
        idx = torch.randperm(n_pixels)[:n_masked]
        frames = frames.clone().reshape(T, -1)
        frames[:, idx] = self.mask_fill_value
        return frames.reshape(T, H, W)

    def _speed_perturb(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate speed change by dropping or duplicating frames
        at regular intervals. Avoids interpolation blur.
        Output is always T frames.

        The rate factor is drawn from {0.85, 0.9, 1.0, 1.1, 1.15}; 1.0 is a
        no-op. Factors < 1.0 resample to fewer frames, then duplicate evenly
        spaced frames to pad back to T; factors > 1.0 resample to more frames,
        then drop evenly spaced frames to trim back to T. Examples (T=75):

        0.85x: resample to 64 frames, then duplicate 11 evenly spaced -> 75
        0.9x : resample to 68 frames, then duplicate 7  evenly spaced -> 75
        1.1x : resample to 82 frames, then drop      7  evenly spaced -> 75
        1.15x: resample to 86 frames, then drop      11 evenly spaced -> 75
        """
        T, H, W = frames.shape
        factor = random.choice([0.85, 0.9, 1.0, 1.1, 1.15])

        if factor == 1.0:
            return frames

        new_T = round(T * factor)

        # resample to new_T via nearest neighbour (no interpolation blur)
        indices = torch.linspace(0, T - 1, new_T).round().long().clamp(0, T - 1)
        frames = frames[indices]  # (new_T, H, W)

        if new_T < T:
            n_dup = T - new_T
            dup_positions = torch.linspace(0, new_T - 1, n_dup).round().long()
            insert_frames = frames[dup_positions]  # (n_dup, H, W)
            frames = torch.cat([frames, insert_frames], dim=0)  # (T, H, W)
            # sort is not strictly necessary but keeps temporal order coherent
            order = torch.cat(
                [
                    torch.arange(new_T),
                    dup_positions,
                ]
            ).argsort(stable=True)
            frames = frames[order]

        elif new_T > T:
            n_drop = new_T - T
            drop_positions = torch.linspace(0, new_T - 1, n_drop).round().long()
            keep_mask = torch.ones(new_T, dtype=torch.bool)
            keep_mask[drop_positions] = False
            frames = frames[keep_mask]  # (T, H, W)

        assert frames.shape[0] == T, f"Expected {T} frames, got {frames.shape[0]}"
        return frames

    def _gaussian_noise(self, frames: torch.Tensor) -> torch.Tensor:
        return frames + torch.randn_like(frames) * self.noise_std

    def _landmarks_to_roi(
        self,
        landmarks: np.ndarray,  # (T, 68, 2) or (68, 2) full face coords
        mouth_w: int,  # same values used in _extract_mouth_frames
        mouth_h: int,
        roi_size: tuple,  # (88, 88)
    ) -> np.ndarray:
        """
        Transform full-face landmark coordinates into the cropped+resized
        ROI pixel space that the augmented frames live in.
        """
        lm = landmarks  # (T, 68, 2) or (68, 2)

        cx = (lm[..., 48, 0] + lm[..., 54, 0]) / 2  # (...,)
        cy = (lm[..., 48, 1] + lm[..., 54, 1]) / 2

        # top-left corner of crop box (before resize), same logic as extraction
        x1 = cx - mouth_w // 2  # (...,)
        y1 = cy - mouth_h // 2

        scale_x = roi_size[0] / mouth_w
        scale_y = roi_size[1] / mouth_h

        roi_lm = lm.copy().astype(float)
        roi_lm[..., 0] = (lm[..., 0] - x1[..., None]) * scale_x
        roi_lm[..., 1] = (lm[..., 1] - y1[..., None]) * scale_y

        return roi_lm  # same shape as input, now in ROI pixel space

    def _mustache_occlusion(
        self,
        frames: torch.Tensor,
        landmarks: np.ndarray,  # (T, 68, 2) already in ROI pixel space
    ) -> torch.Tensor:
        """
        Renders a curved moustache shape anchored to dlib lip landmarks.
        Uses a quadratic bezier curve to define the upper boundary,
        giving a natural arch shape rather than a rectangular patch.
        """
        T, H, W = frames.shape
        frames = frames.clone()

        # stable anchors via median across frames to suppress dlib jitter
        upper_lip_row = int(np.median(landmarks[:, 51, 1]))  # cupid bow centre
        left_col = int(np.median(landmarks[:, 48, 0]))  # left lip corner
        right_col = int(np.median(landmarks[:, 54, 0]))  # right lip corner

        if not (0 < upper_lip_row < H):
            return frames

        # slight random vertical offset so it sits just above the lip
        v_offset = random.randint(1, 3)
        bot_row = max(0, upper_lip_row - v_offset)

        # moustache height: how thick it is vertically
        moustache_h = random.randint(5, 9)
        top_row_centre = max(0, bot_row - moustache_h)

        # narrow inside lip corners
        margin = random.randint(2, 4)
        left_col = max(0, left_col + margin)
        right_col = min(W, right_col - margin)

        if left_col >= right_col or top_row_centre >= bot_row:
            return frames

        # ------------------------------------------------------------------
        # Build a moustache mask using a filled bezier polygon
        # Upper boundary: quadratic bezier curving upward in the centre
        # Lower boundary: straight line along bot_row
        # ------------------------------------------------------------------
        mid_col = (left_col + right_col) // 2

        # control point for upper bezier: arches upward by arch_height pixels
        arch_height = random.randint(3, 6)
        p0 = np.array([left_col, bot_row], dtype=np.float32)
        p1 = np.array([mid_col, top_row_centre - arch_height], dtype=np.float32)
        p2 = np.array([right_col, bot_row], dtype=np.float32)

        n_points = 30
        t_vals = np.linspace(0, 1, n_points)
        bezier_pts = np.array(
            [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2 for t in t_vals]
        ).astype(
            np.int32
        )  # (n_points, 2) in (x, y) / (col, row) order

        # polygon: upper bezier curve + lower straight line closing the shape
        lower_left = np.array([[left_col, bot_row]], dtype=np.int32)
        lower_right = np.array([[right_col, bot_row]], dtype=np.int32)
        polygon = np.concatenate(
            [
                bezier_pts,
                lower_right,
                lower_left,
            ],
            axis=0,
        )  # (n_points+2, 2)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], color=255)
        mask_bool = torch.from_numpy(mask).bool()  # (H, W)

        # ------------------------------------------------------------------
        # Dark grey fill in normalised space
        # the data range is approx [-1.6, 1.3]
        # dark grey sits around -0.5 to -0.8 (not as dark as -1.2 which is black)
        # ------------------------------------------------------------------
        fill_base = random.uniform(-0.8, -0.5)
        noise = torch.randn(T, H, W, dtype=frames.dtype) * 0.04

        fill_tensor = torch.full((T, H, W), fill_base, dtype=frames.dtype) + noise

        mask_expanded = mask_bool.unsqueeze(0).expand(T, -1, -1)  # (T, H, W)
        frames[mask_expanded] = fill_tensor[mask_expanded]

        return frames

    def __call__(self, frames: torch.Tensor, landmarks: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: (T, H, W) float tensor, mean/std normalised, approx range [-1.6, 1.3]
            landmarks: (T, 68, 2) or (68, 2) xy coordinates in the ROI pixel space
        """

        # Occlusion first: the landmarks are in the original ROI pixel space,
        # so the moustache must be rendered before any geometric transform
        # (flip / rotate / crop-resize) moves the frames out of that space.
        if landmarks is not None and torch.rand(1).item() < self.mustache_prob:
            frames = self._mustache_occlusion(frames, landmarks)

        if torch.rand(1).item() < self.speed_perturb_prob:
            frames = self._speed_perturb(frames)

        if torch.rand(1).item() < self.flip_prob:
            frames = self._horizontal_flip(frames)

        if torch.rand(1).item() < self.rotation_prob:
            frames = self._rotate(frames)

        frames = self._random_crop_resize(frames)

        if torch.rand(1).item() < self.blur_prob:
            frames = self._gaussian_blur(frames)

        if torch.rand(1).item() < self.time_mask_prob:
            frames = self._time_masking(frames)

        if torch.rand(1).item() < self.spatial_mask_prob:
            frames = self._spatial_mask(frames)

        if torch.rand(1).item() < self.noise_prob:
            frames = self._gaussian_noise(frames)

        return frames


class VisualInputMode(str, Enum):
    PRECOMPUTED = "precomputed"
    ON_THE_FLY = "on_the_fly"


class VisualFeatureInputStrategy(BatchIO):
    def __init__(
        self,
        frame_shift: float = 0.04,
        mode: str = VisualInputMode.PRECOMPUTED,
        augment=None,
        roi_mode: str = "centroid",
    ):
        super().__init__()
        self.frame_shift = frame_shift
        self.mode = mode
        self.augment = augment
        self.roi_mode = roi_mode

    def _load_precomputed_features(self, cut):
        return torch.from_numpy(cut.load_custom("video_features")).float()

    # AV-HuBERT's own image normalisation (avhubert.utils.Compose of
    # Normalize(0,255) then Normalize(image_mean, image_std)); the constants
    # come from the pretrained task config. Reproduced here so the mean-face
    # path does not have to load the 100M-parameter model just to normalise.
    _IMAGE_MEAN = 0.421
    _IMAGE_STD = 0.165

    def _load_mouth_frames(self, cut):
        video_path = Path(cut.recording.sources[0].source)

        if self.roi_mode == "meanface":
            # Mean-face ROIs are cached raw (uint8), unlike .mouth_frames.npz
            # which already holds transform()ed floats, so normalise here,
            # otherwise the encoder sees 0..255 inputs and silently degrades.
            roi = np.load(video_path.with_suffix(".mfroi.npz"))["roi"]
            frames = (
                roi.astype(np.float32) / 255.0 - self._IMAGE_MEAN
            ) / self._IMAGE_STD
        else:
            frames = np.load(video_path.with_suffix(".mouth_frames.npz"))["frames"]

        frames = torch.FloatTensor(frames)

        if self.augment is not None:
            if self.roi_mode == "meanface":
                # _landmarks_to_roi assumes the centroid crop geometry, which
                # does not hold after the mean-face warp. Landmarks are only
                # consumed by the moustache occlusion (mustache_prob, default
                # 0.0), so None is safe and every other augmentation applies
                # unchanged. Revisit if mustache_prob is ever turned on.
                roi_landmarks = None
            else:
                landmarks = np.load(video_path.with_suffix(".landmarks.npz"))[
                    "landmarks"
                ]
                roi_landmarks = self.augment._landmarks_to_roi(
                    landmarks,
                    mouth_w=64,  # same values used during preprocessing
                    mouth_h=64,
                    roi_size=(88, 88),
                )

            frames = self.augment(frames, roi_landmarks)

        return frames

    def __call__(self, cuts):
        if self.mode == VisualInputMode.PRECOMPUTED:
            inputs = [self._load_precomputed_features(cut) for cut in cuts]

        elif self.mode == VisualInputMode.ON_THE_FLY:
            inputs = [self._load_mouth_frames(cut) for cut in cuts]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        lengths = torch.tensor(
            [x.shape[0] for x in inputs],
            dtype=torch.int32,
        )

        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs,
            batch_first=True,
        )

        return inputs, lengths

    @property
    def extractor(self):
        class DummyExtractor:
            def __init__(self, frame_shift):
                self.frame_shift = frame_shift

        return DummyExtractor(self.frame_shift)

    def supervision_intervals(self, cuts: CutSet) -> Dict[str, torch.Tensor]:
        start_frames, nums_frames, sequence_idx = [], [], []
        for i, cut in enumerate(cuts):
            for sup in cut.supervisions:
                start, num = supervision_to_frames(
                    sup, self.frame_shift, cut.sampling_rate, max_frames=None
                )
                start_frames.append(start)
                nums_frames.append(num)
                sequence_idx.append(i)
        return {
            "sequence_idx": torch.tensor(sequence_idx, dtype=torch.int32),
            "start_frame": torch.tensor(start_frames, dtype=torch.int32),
            "num_frames": torch.tensor(nums_frames, dtype=torch.int32),
        }
