# Modified from egs/ljspeech/TTS/vits/loss.py by: Zengrui JIN (Tsinghua University)
# original implementation is from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/hifigan/loss.py

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encodec-related loss modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class GeneratorAdversarialLoss(torch.nn.Module):
    """Generator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "hinge",
    ):
        """Initialize GeneratorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(
        self,
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Calcualate generator adversarial loss.

        Args:
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs..

        Returns:
            Tensor: Generator adversarial loss value.

        """
        adv_loss = 0.0
        if isinstance(outputs, (tuple, list)):
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_ = outputs_[-1]
                adv_loss += self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            for i, outputs_ in enumerate(outputs):
                adv_loss += self.criterion(outputs_)
            adv_loss /= i + 1
        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return F.relu(1 - x).mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):
    """Discriminator adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators: bool = True,
        loss_type: str = "hinge",
    ):
        """Initialize DiscriminatorAversarialLoss module.

        Args:
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            loss_type (str): Loss type, "mse" or "hinge".

        """
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(
        self,
        outputs_hat: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
        outputs: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from generator.
            outputs (Union[List[List[Tensor]], List[Tensor], Tensor]): Discriminator
                outputs, list of discriminator outputs, or list of list of discriminator
                outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        real_loss = 0.0
        fake_loss = 0.0
        if isinstance(outputs, (tuple, list)):
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    # NOTE(kan-bayashi): case including feature maps
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                real_loss += self.real_criterion(outputs_)
                fake_loss += self.fake_criterion(outputs_hat_)
            fake_loss /= i + 1
            real_loss /= i + 1

        return real_loss, fake_loss

    def _mse_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.ones_like(x) - x).mean()

    def _hinge_fake_loss(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(torch.ones_like(x) + x).mean()


class FeatureLoss(torch.nn.Module):
    """Feature loss module."""

    def __init__(
        self,
        average_by_layers: bool = True,
        average_by_discriminators: bool = True,
        include_final_outputs: bool = True,
    ):
        """Initialize FeatureMatchLoss module.

        Args:
            average_by_layers (bool): Whether to average the loss by the number
                of layers.
            average_by_discriminators (bool): Whether to average the loss by
                the number of discriminators.
            include_final_outputs (bool): Whether to include the final output of
                each discriminator for loss calculation.

        """
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(
        self,
        feats_hat: Union[List[List[torch.Tensor]], List[torch.Tensor]],
        feats: Union[List[List[torch.Tensor]], List[torch.Tensor]],
    ) -> torch.Tensor:
        """Calculate feature matching loss.

        Args:
            feats_hat (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from generator's outputs.
            feats (Union[List[List[Tensor]], List[Tensor]]): List of list of
                discriminator outputs or list of discriminator outputs calcuated
                from groundtruth..

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += (
                    F.l1_loss(feat_hat_, feat_.detach()) / (feat_.detach().abs().mean())
                ).mean()
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1

        return feat_match_loss


class MelSpectrogramReconstructionLoss(torch.nn.Module):
    """Mel Spec Reconstruction loss."""

    def __init__(
        self,
        sampling_rate: int = 22050,
        n_mels: int = 64,
        use_fft_mag: bool = True,
        return_mel: bool = False,
    ):
        super().__init__()
        self.wav_to_specs = []
        for i in range(5, 12):
            s = 2**i
            self.wav_to_specs.append(
                MelSpectrogram(
                    sample_rate=sampling_rate,
                    n_fft=max(s, 512),
                    win_length=s,
                    hop_length=s // 4,
                    n_mels=n_mels,
                )
            )
        self.return_mel = return_mel

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Calculate Mel-spectrogram loss.

        Args:
            x_hat (Tensor): Generated waveform tensor (B, 1, T).
            x (Tensor): Groundtruth waveform tensor (B, 1, T).
            spec (Optional[Tensor]): Groundtruth linear amplitude spectrum tensor
                (B, T, n_fft // 2 + 1).  if provided, use it instead of groundtruth
                waveform.

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_loss = 0.0

        for i, wav_to_spec in enumerate(self.wav_to_specs):
            s = 2 ** (i + 5)
            wav_to_spec.to(x.device)

            mel_hat = wav_to_spec(x_hat.squeeze(1))
            mel = wav_to_spec(x.squeeze(1))

            mel_loss += (
                F.l1_loss(mel_hat, mel, reduce=True, reduction="mean")
                + (
                    (
                        (torch.log(mel.abs() + 1e-7) - torch.log(mel_hat.abs() + 1e-7))
                        ** 2
                    ).mean(dim=-2)
                    ** 0.5
                ).mean()
            )

        # mel_hat = self.wav_to_spec(x_hat.squeeze(1))
        # mel = self.wav_to_spec(x.squeeze(1))
        # mel_loss = F.l1_loss(mel_hat, mel) + F.mse_loss(mel_hat, mel)

        if self.return_mel:
            return mel_loss, (mel_hat, mel)

        return mel_loss


class WavReconstructionLoss(torch.nn.Module):
    """Wav Reconstruction loss."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate wav loss.

        Args:
            x_hat (Tensor): Generated waveform tensor (B, 1, T).
            x (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Wav loss value.

        """
        wav_loss = F.l1_loss(x, x_hat)

        return wav_loss
