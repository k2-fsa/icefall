import math
import random
from typing import List

import numpy as np
import torch
from loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
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
        multi_period_discriminator: nn.Module,
        multi_scale_stft_discriminator: nn.Module,
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
        self.feature_match_loss = FeatureMatchLoss(average_by_layers=False)
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
            y_p, y_p_hat, fmap_p, fmap_p_hat = self.multi_period_discriminator(
                speech.contiguous(),
                speech_hat.contiguous(),
            )
            y_s, y_s_hat, fmap_s, fmap_s_hat = self.multi_scale_discriminator(
                speech.contiguous(),
                speech_hat.contiguous(),
            )

        # calculate losses
        with autocast(enabled=False):
            gen_stft_adv_loss = self.generator_adversarial_loss(outputs=y_hat)
            gen_period_adv_loss = self.generator_adversarial_loss(outputs=y_p_hat)
            gen_scale_adv_loss = self.generator_adversarial_loss(outputs=y_s_hat)

            feature_stft_loss = self.feature_match_loss(feats=fmap, feats_hat=fmap_hat)
            feature_period_loss = self.feature_match_loss(
                feats=fmap_p, feats_hat=fmap_p_hat
            )
            feature_scale_loss = self.feature_match_loss(
                feats=fmap_s, feats_hat=fmap_s_hat
            )

            wav_reconstruction_loss = self.wav_reconstruction_loss(
                x=speech, x_hat=speech_hat
            )
            mel_reconstruction_loss = self.mel_reconstruction_loss(
                x=speech, x_hat=speech_hat
            )

            # loss, rec_loss, adv_loss, feat_loss, d_weight = loss_g(
            #     commit_loss,
            #     speech,
            #     speech_hat,
            #     fmap,
            #     fmap_hat,
            #     y,
            #     y_hat,
            #     y_p,
            #     y_p_hat,
            #     y_s,
            #     y_s_hat,
            #     fmap_p,
            #     fmap_p_hat,
            #     fmap_s,
            #     fmap_s_hat,
            #     args=self.params,
            # )

        stats = dict(
            # generator_loss=loss.item(),
            generator_wav_reconstruction_loss=wav_reconstruction_loss.item(),
            generator_mel_reconstruction_loss=mel_reconstruction_loss.item(),
            generator_feature_stft_loss=feature_stft_loss.item(),
            generator_feature_period_loss=feature_period_loss.item(),
            generator_feature_scale_loss=feature_scale_loss.item(),
            generator_stft_adv_loss=gen_stft_adv_loss.item(),
            generator_period_adv_loss=gen_period_adv_loss.item(),
            generator_scale_adv_loss=gen_scale_adv_loss.item(),
            generator_commit_loss=commit_loss.item(),
            # d_weight=d_weight.item(),
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
        y_p, y_p_hat, fmap_p, fmap_p_hat = self.multi_period_discriminator(
            speech.contiguous(),
            speech_hat.contiguous().detach(),
        )
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
            (
                disc_period_real_adv_loss,
                disc_period_fake_adv_loss,
            ) = self.discriminator_adversarial_loss(outputs=y_p, outputs_hat=y_p_hat)
            (
                disc_scale_real_adv_loss,
                disc_scale_fake_adv_loss,
            ) = self.discriminator_adversarial_loss(outputs=y_s, outputs_hat=y_s_hat)

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
