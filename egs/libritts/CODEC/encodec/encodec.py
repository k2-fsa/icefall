import math
import random
from typing import List

import numpy as np
import torch
from loss import loss_dis, loss_g
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

    def _forward_generator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        global_step: int,
        return_sample: bool = False,
    ):
        """Perform generator forward.

        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            global_step (int): Global step.
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
                random.randint(0, len(self.target_bandwidths) - 1), device=speech.device,
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
            loss, rec_loss, adv_loss, feat_loss, d_weight = loss_g(
                commit_loss,
                speech,
                speech_hat,
                fmap,
                fmap_hat,
                y,
                y_hat,
                global_step,
                y_p,
                y_p_hat,
                y_s,
                y_s_hat,
                fmap_p,
                fmap_p_hat,
                fmap_s,
                fmap_s_hat,
                args=self.params,
            )

        stats = dict(
            generator_loss=loss.item(),
            generator_reconstruction_loss=rec_loss.item(),
            generator_feature_loss=feat_loss.item(),
            generator_adv_loss=adv_loss.item(),
            generator_commit_loss=commit_loss.item(),
            d_weight=d_weight.item(),
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
        return loss, stats

    def _forward_discriminator(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        global_step: int,
    ):
        """
        Args:
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            global_step (int): Global step.

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
                random.randint(0, len(self.target_bandwidths) - 1), device=speech.device,
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
            loss = loss_dis(
                y,
                y_hat,
                fmap,
                fmap_hat,
                y_p,
                y_p_hat,
                fmap_p,
                fmap_p_hat,
                y_s,
                y_s_hat,
                fmap_s,
                fmap_s_hat,
                global_step,
                args=self.params,
            )
        stats = dict(
            discriminator_loss=loss.item(),
        )

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return loss, stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        global_step: int,
        return_sample: bool,
        forward_generator: bool,
    ):
        if forward_generator:
            return self._forward_generator(
                speech=speech,
                speech_lengths=speech_lengths,
                global_step=global_step,
                return_sample=return_sample,
            )
        else:
            return self._forward_discriminator(
                speech=speech,
                speech_lengths=speech_lengths,
                global_step=global_step,
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
        o = self.decoder(quantized)
        return o

    def inference(self, x, target_bw=None, st=None):
        # setup
        x = x.unsqueeze(1)

        codes = self.encode(x, target_bw, st)
        o = self.decode(codes)
        return o
