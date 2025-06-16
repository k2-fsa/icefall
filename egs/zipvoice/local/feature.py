#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
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

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames


class MelSpectrogramFeatures(nn.Module):
    def __init__(
        self,
        sampling_rate=24000,
        n_mels=100,
        n_fft=1024,
        hop_length=256,
    ):
        super().__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def forward(self, inp):
        assert len(inp.shape) == 2

        mel = self.mel_spec(inp)
        logmel = mel.clamp(min=1e-7).log()
        return logmel


@dataclass
class TorchAudioFbankConfig:
    sampling_rate: int
    n_mels: int
    n_fft: int
    hop_length: int


@register_extractor
class TorchAudioFbank(FeatureExtractor):

    name = "TorchAudioFbank"
    config_type = TorchAudioFbankConfig

    def __init__(self, config):
        super().__init__(config=config)

    def _feature_fn(self, sample):
        fbank = MelSpectrogramFeatures(
            sampling_rate=self.config.sampling_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        return fbank(sample)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor]:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        assert samples.ndim == 2, samples.shape
        assert samples.shape[0] == 1, samples.shape

        mel = self._feature_fn(samples).squeeze().t()

        assert mel.ndim == 2, mel.shape
        assert mel.shape[1] == self.config.n_mels, mel.shape

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        if is_numpy:
            return mel.cpu().numpy()
        else:
            return mel

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate
