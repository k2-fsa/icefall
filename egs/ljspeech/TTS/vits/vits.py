# based on https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/vits/vits.py

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS module for GAN-TTS task."""

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from generator import VITSGenerator
from hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)
from loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    KLDivergenceLoss,
    MelSpectrogramLoss,
)
from torch.cuda.amp import autocast
from utils import get_segments

AVAILABLE_GENERATERS = {
    "vits_generator": VITSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
}

LOW_CONFIG = {
    "hidden_channels": 96,
    "decoder_upsample_scales": (8, 8, 4),
    "decoder_channels": 256,
    "decoder_upsample_kernel_sizes": (16, 16, 8),
    "decoder_resblock_kernel_sizes": (3, 5, 7),
    "decoder_resblock_dilations": ((1, 2), (2, 6), (3, 12)),
    "text_encoder_cnn_module_kernel": 3,
}

MEDIUM_CONFIG = {
    "hidden_channels": 192,
    "decoder_upsample_scales": (8, 8, 4),
    "decoder_channels": 256,
    "decoder_upsample_kernel_sizes": (16, 16, 8),
    "decoder_resblock_kernel_sizes": (3, 5, 7),
    "decoder_resblock_dilations": ((1, 2), (2, 6), (3, 12)),
    "text_encoder_cnn_module_kernel": 3,
}

HIGH_CONFIG = {
    "hidden_channels": 192,
    "decoder_upsample_scales": (8, 8, 2, 2),
    "decoder_channels": 512,
    "decoder_upsample_kernel_sizes": (16, 16, 4, 4),
    "decoder_resblock_kernel_sizes": (3, 7, 11),
    "decoder_resblock_dilations": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    "text_encoder_cnn_module_kernel": 5,
}


class VITS(nn.Module):
    """Implement VITS, `Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`"""

    def __init__(
        self,
        # generator related
        vocab_size: int,
        feature_dim: int = 513,
        sampling_rate: int = 22050,
        generator_type: str = "vits_generator",
        model_type: str = "",
        generator_params: Dict[str, Any] = {
            "hidden_channels": 192,
            "spks": None,
            "langs": None,
            "spk_embed_dim": None,
            "global_channels": -1,
            "segment_size": 32,
            "text_encoder_attention_heads": 2,
            "text_encoder_ffn_expand": 4,
            "text_encoder_cnn_module_kernel": 5,
            "text_encoder_blocks": 6,
            "text_encoder_dropout_rate": 0.1,
            "decoder_kernel_size": 7,
            "decoder_channels": 512,
            "decoder_upsample_scales": [8, 8, 2, 2],
            "decoder_upsample_kernel_sizes": [16, 16, 4, 4],
            "decoder_resblock_kernel_sizes": [3, 7, 11],
            "decoder_resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_weight_norm_in_decoder": True,
            "posterior_encoder_kernel_size": 5,
            "posterior_encoder_layers": 16,
            "posterior_encoder_stacks": 1,
            "posterior_encoder_base_dilation": 1,
            "posterior_encoder_dropout_rate": 0.0,
            "use_weight_norm_in_posterior_encoder": True,
            "flow_flows": 4,
            "flow_kernel_size": 5,
            "flow_base_dilation": 1,
            "flow_layers": 4,
            "flow_dropout_rate": 0.0,
            "use_weight_norm_in_flow": True,
            "use_only_mean_in_flow": True,
            "stochastic_duration_predictor_kernel_size": 3,
            "stochastic_duration_predictor_dropout_rate": 0.5,
            "stochastic_duration_predictor_flows": 4,
            "stochastic_duration_predictor_dds_conv_layers": 3,
        },
        # discriminator related
        discriminator_type: str = "hifigan_multi_scale_multi_period_discriminator",
        discriminator_params: Dict[str, Any] = {
            "scales": 1,
            "scale_downsample_pooling": "AvgPool1d",
            "scale_downsample_pooling_params": {
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            "scale_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "follow_official_norm": False,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        mel_loss_params: Dict[str, Any] = {
            "frame_shift": 256,
            "frame_length": 1024,
            "n_mels": 80,
        },
        lambda_adv: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_feat_match: float = 2.0,
        lambda_dur: float = 1.0,
        lambda_kl: float = 1.0,
        cache_generator_outputs: bool = True,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Input vocabulary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since VITS is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            sampling_rate (int): Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            model_type (str): If not empty, must be one of: low, medium, high
            generator_type (str): Generator type.
            generator_params (Dict[str, Any]): Parameter dict for generator.
            discriminator_type (str): Discriminator type.
            discriminator_params (Dict[str, Any]): Parameter dict for discriminator.
            generator_adv_loss_params (Dict[str, Any]): Parameter dict for generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameter dict for
                discriminator adversarial loss.
            feat_match_loss_params (Dict[str, Any]): Parameter dict for feat match loss.
            mel_loss_params (Dict[str, Any]): Parameter dict for mel loss.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_mel (float): Loss scaling coefficient for mel spectrogram loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_kl (float): Loss scaling coefficient for KL divergence loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        super().__init__()

        generator_params = copy.deepcopy(generator_params)
        discriminator_params = copy.deepcopy(discriminator_params)
        generator_adv_loss_params = copy.deepcopy(generator_adv_loss_params)
        discriminator_adv_loss_params = copy.deepcopy(discriminator_adv_loss_params)
        feat_match_loss_params = copy.deepcopy(feat_match_loss_params)
        mel_loss_params = copy.deepcopy(mel_loss_params)

        if model_type != "":
            assert model_type in ("low", "medium", "high"), model_type
            if model_type == "low":
                generator_params.update(LOW_CONFIG)
            elif model_type == "medium":
                generator_params.update(MEDIUM_CONFIG)
            elif model_type == "high":
                generator_params.update(HIGH_CONFIG)
            else:
                raise ValueError(f"Unknown model_type: ${model_type}")

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        if generator_type == "vits_generator":
            # NOTE(kan-bayashi): Update parameters for the compatibility.
            #   The idim and odim is automatically decided from input data,
            #   where idim represents #vocabularies and odim represents
            #   the input acoustic feature dimension.
            generator_params.update(vocabs=vocab_size, aux_channels=feature_dim)
        self.generator = generator_class(
            **generator_params,
        )
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        self.discriminator = discriminator_class(
            **discriminator_params,
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.feat_match_loss = FeatureMatchLoss(
            **feat_match_loss_params,
        )
        mel_loss_params.update(sampling_rate=sampling_rate)
        self.mel_loss = MelSpectrogramLoss(
            **mel_loss_params,
        )
        self.kl_loss = KLDivergenceLoss()

        # coefficients
        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.sampling_rate = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator.spks
        self.langs = self.generator.langs
        self.spk_embed_dim = self.generator.spk_embed_dim

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_sample: bool = False,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            - loss (Tensor): Loss scalar tensor.
            - stats (Dict[str, float]): Statistics to be monitored.
        """
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                return_sample=return_sample,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )

    def _forward_generator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        return_sample: bool = False,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            * loss (Tensor): Loss scalar tensor.
            * stats (Dict[str, float]): Statistics to be monitored.
        """
        # setup
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
        _, z_p, m_p, logs_p, _, logs_q = outs_
        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(speech_)

        # calculate losses
        with autocast(enabled=False):
            if not return_sample:
                mel_loss = self.mel_loss(speech_hat_, speech_)
            else:
                mel_loss, (mel_hat_, mel_) = self.mel_loss(
                    speech_hat_, speech_, return_mel=True
                )
            kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            dur_loss = torch.sum(dur_nll.float())
            adv_loss = self.generator_adv_loss(p_hat)
            feat_match_loss = self.feat_match_loss(p_hat, p)

            mel_loss = mel_loss * self.lambda_mel
            kl_loss = kl_loss * self.lambda_kl
            dur_loss = dur_loss * self.lambda_dur
            adv_loss = adv_loss * self.lambda_adv
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_mel_loss=mel_loss.item(),
            generator_kl_loss=kl_loss.item(),
            generator_dur_loss=dur_loss.item(),
            generator_adv_loss=adv_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
        )

        if return_sample:
            stats["returned_sample"] = (
                speech_hat_[0].data.cpu().numpy(),
                speech_[0].data.cpu().numpy(),
                mel_hat_[0].data.cpu().numpy(),
                mel_[0].data.cpu().numpy(),
            )

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return loss, stats

    def _forward_discrminator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            * loss (Tensor): Loss scalar tensor.
            * stats (Dict[str, float]): Statistics to be monitored.
        """
        # setup
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, _, _, start_idxs, *_ = outs
        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_.detach())
        p = self.discriminator(speech_)

        # calculate losses
        with autocast(enabled=False):
            real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
            loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            discriminator_real_loss=real_loss.item(),
            discriminator_fake_loss=fake_loss.item(),
        )

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return loss, stats

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference for single sample.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            feats (Tensor): Feature tensor (T_feats, aux_channels).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            durations (Tensor): Ground-truth duration tensor (T_text,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            * wav (Tensor): Generated waveform tensor (T_wav,).
            * att_w (Tensor): Monotonic attention weight tensor (T_feats, T_text).
            * duration (Tensor): Predicted duration tensor (T_text,).
        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        if sids is not None:
            sids = sids.view(1)
        if lids is not None:
            lids = lids.view(1)
        if durations is not None:
            durations = durations.view(1, 1, -1)

        # inference
        if use_teacher_forcing:
            assert feats is not None
            feats = feats[None].transpose(1, 2)
            feats_lengths = torch.tensor(
                [feats.size(2)],
                dtype=torch.long,
                device=feats.device,
            )
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                max_len=max_len,
                use_teacher_forcing=use_teacher_forcing,
            )
        else:
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                dur=durations,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
            )
        return wav.view(-1), att_w[0], dur[0]

    def inference_batch(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference for one batch.

        Args:
            text (Tensor): Input text index tensor (B, T_text).
            text_lengths (Tensor): Input text index tensor (B,).
            sids (Tensor): Speaker index tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Tensor): Language index tensor (B,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length.

        Returns:
            * wav (Tensor): Generated waveform tensor (B, T_wav).
            * att_w (Tensor): Monotonic attention weight tensor (B, T_feats, T_text).
            * duration (Tensor): Predicted duration tensor (B, T_text).
        """
        # inference
        wav, att_w, dur = self.generator.inference(
            text=text,
            text_lengths=text_lengths,
            sids=sids,
            spembs=spembs,
            lids=lids,
            noise_scale=noise_scale,
            noise_scale_dur=noise_scale_dur,
            alpha=alpha,
            max_len=max_len,
        )
        return wav, att_w, dur
