#!/usr/bin/env python3
# Copyright         2024 The Chinese University of HK   (Author: Zengrui Jin)
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

import math
import random
from typing import List, Optional

import numpy as np
import torch
from loss import (
    DiscriminatorAdversarialLoss,
    FeatureLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramReconstructionLoss,
    WavReconstructionLoss,
)
from torch import nn
from torch.cuda.amp import autocast


class Encodec(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        target_bandwidths: List[float],
        params: dict,
        encoder: nn.Module,
        quantizer: nn.Module,
        decoder: nn.Module,
        multi_scale_discriminator: nn.Module,
        multi_period_discriminator: Optional[nn.Module] = None,
        multi_scale_stft_discriminator: Optional[nn.Module] = None,
        cache_generator_outputs: bool = False,
    ):
        super(Encodec, self).__init__()

        self.params = params

        # setup the generator
        self.sampling_rate = sampling_rate
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

        self.ratios = encoder.ratios
        self.hop_length = np.prod(self.ratios)
        self.frame_rate = math.ceil(self.sampling_rate / np.prod(self.ratios))
        self.target_bandwidths = target_bandwidths

        # discriminators
        self.multi_scale_discriminator = multi_scale_discriminator
        self.multi_period_discriminator = multi_period_discriminator
        self.multi_scale_stft_discriminator = multi_scale_stft_discriminator

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # construct loss functions
        self.generator_adversarial_loss = GeneratorAdversarialLoss(
            average_by_discriminators=True, loss_type="hinge"
        )
        self.discriminator_adversarial_loss = DiscriminatorAdversarialLoss(
            average_by_discriminators=True, loss_type="hinge"
        )
        self.feature_match_loss = FeatureLoss()
        self.wav_reconstruction_loss = WavReconstructionLoss()
        self.mel_reconstruction_loss = MelSpectrogramReconstructionLoss(
            sampling_rate=self.sampling_rate
        )

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_sample: bool = False,
    ):
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            return_sample (bool): Return the generator output.

        Returns:
            * loss (Tensor): Loss scalar tensor.
            * stats (Dict[str, float]): Statistics to be monitored.
        """
        # setup
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            e = self.encoder(speech)
            index = torch.tensor(
                random.randint(0, len(self.target_bandwidths) - 1),
                device=speech.device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)
            bw = self.target_bandwidths[index.item()]
            quantized, codes, bandwidth, commit_loss = self.quantizer(
                e, self.frame_rate, bw
            )
            speech_hat = self.decoder(quantized)
        else:
            speech_hat = self._cache
        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = speech_hat

        # calculate discriminator outputs
        y_hat, fmap_hat = self.multi_scale_stft_discriminator(speech_hat.contiguous())
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            y, fmap = self.multi_scale_stft_discriminator(speech.contiguous())

            gen_period_adv_loss = torch.tensor(0.0)
            feature_period_loss = torch.tensor(0.0)
            if self.multi_period_discriminator is not None:
                y_p, y_p_hat, fmap_p, fmap_p_hat = self.multi_period_discriminator(
                    speech.contiguous(),
                    speech_hat.contiguous(),
                )

            gen_scale_adv_loss = torch.tensor(0.0)
            feature_scale_loss = torch.tensor(0.0)
            if self.multi_scale_discriminator is not None:
                y_s, y_s_hat, fmap_s, fmap_s_hat = self.multi_scale_discriminator(
                    speech.contiguous(),
                    speech_hat.contiguous(),
                )

        # calculate losses
        with autocast(enabled=False):
            gen_stft_adv_loss = self.generator_adversarial_loss(outputs=y_hat)

            if self.multi_period_discriminator is not None:
                gen_period_adv_loss = self.generator_adversarial_loss(outputs=y_p_hat)
            if self.multi_scale_discriminator is not None:
                gen_scale_adv_loss = self.generator_adversarial_loss(outputs=y_s_hat)

            feature_stft_loss = self.feature_match_loss(feats=fmap, feats_hat=fmap_hat)

            if self.multi_period_discriminator is not None:
                feature_period_loss = self.feature_match_loss(
                    feats=fmap_p, feats_hat=fmap_p_hat
                )
            if self.multi_scale_discriminator is not None:
                feature_scale_loss = self.feature_match_loss(
                    feats=fmap_s, feats_hat=fmap_s_hat
                )

            wav_reconstruction_loss = self.wav_reconstruction_loss(
                x=speech, x_hat=speech_hat
            )
            mel_reconstruction_loss = self.mel_reconstruction_loss(
                x=speech, x_hat=speech_hat
            )

        stats = dict(
            generator_wav_reconstruction_loss=wav_reconstruction_loss.item(),
            generator_mel_reconstruction_loss=mel_reconstruction_loss.item(),
            generator_feature_stft_loss=feature_stft_loss.item(),
            generator_feature_period_loss=feature_period_loss.item(),
            generator_feature_scale_loss=feature_scale_loss.item(),
            generator_stft_adv_loss=gen_stft_adv_loss.item(),
            generator_period_adv_loss=gen_period_adv_loss.item(),
            generator_scale_adv_loss=gen_scale_adv_loss.item(),
            generator_commit_loss=commit_loss.item(),
        )

        if return_sample:
            stats["returned_sample"] = (
                speech_hat.cpu(),
                speech.cpu(),
                fmap_hat[0][0].data.cpu(),
                fmap[0][0].data.cpu(),
            )

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None
        return (
            commit_loss,
            gen_stft_adv_loss,
            gen_period_adv_loss,
            gen_scale_adv_loss,
            feature_stft_loss,
            feature_period_loss,
            feature_scale_loss,
            wav_reconstruction_loss,
            mel_reconstruction_loss,
            stats,
        )

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """
        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).

        Returns:
            * loss (Tensor): Loss scalar tensor.
            * stats (Dict[str, float]): Statistics to be monitored.
        """
        # setup
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            e = self.encoder(speech)
            index = torch.tensor(
                random.randint(0, len(self.target_bandwidths) - 1),
                device=speech.device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(index, src=0)
            bw = self.target_bandwidths[index.item()]
            quantized, codes, bandwidth, commit_loss = self.quantizer(
                e, self.frame_rate, bw
            )
            speech_hat = self.decoder(quantized)
        else:
            speech_hat = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = speech_hat

        # calculate discriminator outputs
        y, fmap = self.multi_scale_stft_discriminator(speech.contiguous())
        y_hat, fmap_hat = self.multi_scale_stft_discriminator(
            speech_hat.contiguous().detach()
        )

        disc_period_real_adv_loss = torch.tensor(0.0)
        disc_period_fake_adv_loss = torch.tensor(0.0)
        if self.multi_period_discriminator is not None:
            y_p, y_p_hat, fmap_p, fmap_p_hat = self.multi_period_discriminator(
                speech.contiguous(),
                speech_hat.contiguous().detach(),
            )

        disc_scale_real_adv_loss = torch.tensor(0.0)
        disc_scale_fake_adv_loss = torch.tensor(0.0)
        if self.multi_scale_discriminator is not None:
            y_s, y_s_hat, fmap_s, fmap_s_hat = self.multi_scale_discriminator(
                speech.contiguous(),
                speech_hat.contiguous().detach(),
            )
        # calculate losses
        with autocast(enabled=False):
            (
                disc_stft_real_adv_loss,
                disc_stft_fake_adv_loss,
            ) = self.discriminator_adversarial_loss(outputs=y, outputs_hat=y_hat)
            if self.multi_period_discriminator is not None:
                (
                    disc_period_real_adv_loss,
                    disc_period_fake_adv_loss,
                ) = self.discriminator_adversarial_loss(
                    outputs=y_p, outputs_hat=y_p_hat
                )
            if self.multi_scale_discriminator is not None:
                (
                    disc_scale_real_adv_loss,
                    disc_scale_fake_adv_loss,
                ) = self.discriminator_adversarial_loss(
                    outputs=y_s, outputs_hat=y_s_hat
                )

        stats = dict(
            discriminator_stft_real_adv_loss=disc_stft_real_adv_loss.item(),
            discriminator_period_real_adv_loss=disc_period_real_adv_loss.item(),
            discriminator_scale_real_adv_loss=disc_scale_real_adv_loss.item(),
            discriminator_stft_fake_adv_loss=disc_stft_fake_adv_loss.item(),
            discriminator_period_fake_adv_loss=disc_period_fake_adv_loss.item(),
            discriminator_scale_fake_adv_loss=disc_scale_fake_adv_loss.item(),
        )

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return (
            disc_stft_real_adv_loss,
            disc_stft_fake_adv_loss,
            disc_period_real_adv_loss,
            disc_period_fake_adv_loss,
            disc_scale_real_adv_loss,
            disc_scale_fake_adv_loss,
            stats,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_sample: bool,
        forward_generator: bool,
    ):
        if forward_generator:
            return self._forward_generator(
                speech=speech,
                speech_lengths=speech_lengths,
                return_sample=return_sample,
            )
        else:
            return self._forward_discriminator(
                speech=speech,
                speech_lengths=speech_lengths,
            )

    def encode(self, x, target_bw=None, st=None):
        e = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes

    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        x_hat = self.decoder(quantized)
        return x_hat

    def inference(self, x, target_bw=None, st=None):
        # setup
        x = x.unsqueeze(1)

        codes = self.encode(x, target_bw, st)
        x_hat = self.decode(codes)
        return codes, x_hat
