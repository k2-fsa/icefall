from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from audio import mel_spectrogram
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames


@dataclass
class MatchaFbankConfig:
    n_fft: int
    n_mels: int
    sampling_rate: int
    hop_length: int
    win_length: int
    f_min: float
    f_max: float


@register_extractor
class MatchaFbank(FeatureExtractor):

    name = "MatchaFbank"
    config_type = MatchaFbankConfig

    def __init__(self, config):
        super().__init__(config=config)

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: np.ndarray,
        sampling_rate: int,
    ) -> torch.Tensor:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        samples = torch.from_numpy(samples)
        assert samples.ndim == 2, samples.shape
        assert samples.shape[0] == 1, samples.shape

        mel = (
            mel_spectrogram(
                samples,
                self.config.n_fft,
                self.config.n_mels,
                self.config.sampling_rate,
                self.config.hop_length,
                self.config.win_length,
                self.config.f_min,
                self.config.f_max,
                center=False,
            )
            .squeeze()
            .t()
        )

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

        return mel.numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate
