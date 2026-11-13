# Copyright    2024  Yuekai Zhang
#              2026  Shanghai Jiao Tong University  (authors: Yifan Yang)
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

from typing import Optional, Tuple

import torch
from encoder import spear_encode_fbank
from torch import nn
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_SPEECH_TOKEN = "<speech>"


class EncoderProjector(nn.Module):
    """
    Projects the speech encoder outputs to the LLM embedding space.

    It layer-normalizes the encoder outputs, then stacks ``downsample_rate``
    consecutive frames along the feature dimension (reducing the sequence length
    by the same factor) and applies a 2-layer MLP. Modified from
    https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/models/projector.py.

    Args:
        encoder_dim: The dimension of the speech encoder outputs.
        llm_dim: The hidden dimension of the language model.
        downsample_rate: The number of frames to stack together.
        hidden_dim: The hidden dimension of the 2-layer MLP.
    """

    def __init__(self, encoder_dim, llm_dim, downsample_rate=5, hidden_dim=4096):
        super().__init__()
        self.downsample_rate = downsample_rate
        # SALMONN's ln_audio: normalize before stacking.
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.linear1 = nn.Linear(encoder_dim * self.downsample_rate, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, llm_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.downsample_rate, feat_dim * self.downsample_rate
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SPEECH_LLM(nn.Module):
    """
    The Speech-to-Text model. It consists of a speech encoder, a language model
    and an encoder projector.

    The encoder is the Zipformer of a SPEAR model, fed the 128-bin fbank that
    the dataloader computes (MUSAN and SpecAugment are applied there too, exactly
    as in the zipformer recipe). The encoder projector maps its output to the LLM
    embedding space, and the language model generates the text conditioned on it.

    Args:
        encoder: The SPEAR speech encoder module (see ``encoder.load_spear_encoder``).
        llm: The language model module.
        encoder_projector: The encoder projector module.
        padding_side: Padding side used for packed training sequences.
    """

    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        padding_side: str = "left",
    ):
        super().__init__()
        if padding_side not in ("left", "right"):
            raise ValueError(f"Unsupported padding_side: {padding_side!r}")
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.padding_side = padding_side

    def _encode_features(
        self, feature: torch.Tensor, feature_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return SPEAR states and their valid lengths before projector stacking."""
        outputs = spear_encode_fbank(
            self.encoder, feature, feature_lens, return_middle_layers=False
        )
        return outputs["encoder_out"], outputs["encoder_out_lens"]

    def _merge_input_ids_with_speech_features(
        self,
        speech_features,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels=None,
        *,
        speech_lens: torch.Tensor,
        padding_side: Optional[str] = None,
    ):
        """Replace each speech token with that utterance's valid speech frames.

        Text padding is removed before packing. Speech padding never enters the
        LLM sequence, and padding is added only around the final packed rows.
        """
        if speech_features.ndim != 3 or inputs_embeds.ndim != 3:
            raise ValueError("Speech and text embeddings must be 3-D tensors")
        batch_size, max_speech_len, hidden_size = speech_features.shape
        if batch_size == 0:
            raise ValueError("Cannot merge an empty batch")
        if (
            inputs_embeds.shape[0] != batch_size
            or inputs_embeds.shape[2] != hidden_size
        ):
            raise ValueError("Speech and text embedding shapes do not match")
        if (
            input_ids.shape != inputs_embeds.shape[:2]
            or attention_mask.shape != input_ids.shape
        ):
            raise ValueError("Text ids, mask and embeddings must have matching shapes")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("Labels must have the same shape as input_ids")
        if speech_lens.shape != (batch_size,) or speech_lens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "speech_lens must be a 1-D integer tensor, one length per utterance"
            )

        device = inputs_embeds.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        speech_lens = speech_lens.to(device=device, dtype=torch.long)
        if bool(((speech_lens < 0) | (speech_lens > max_speech_len)).any()):
            raise ValueError("speech_lens is outside the available speech frames")
        speech_features = speech_features.to(device=device, dtype=inputs_embeds.dtype)
        text_valid = attention_mask.bool()
        speech_tokens = (
            input_ids.eq(self.llm.config.default_speech_token_id) & text_valid
        )
        if not bool(speech_tokens.sum(dim=1).eq(1).all()):
            raise ValueError(
                "Each utterance must contain exactly one unmasked speech token"
            )

        # A regular token contributes one position, a speech token contributes
        # its own frame count, and a padding token contributes none.
        widths = text_valid.long() + speech_tokens.long() * (speech_lens[:, None] - 1)
        ends = widths.cumsum(dim=1) - 1
        lengths = widths.sum(dim=1)
        if bool(lengths.le(0).any()):
            raise ValueError(
                "Every merged utterance must contain at least one position"
            )
        max_length = int(lengths.max())
        side = self.padding_side if padding_side is None else padding_side
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported padding_side: {side!r}")
        offsets = max_length - lengths if side == "left" else torch.zeros_like(lengths)
        ends = ends + offsets[:, None]

        merged = inputs_embeds.new_zeros((batch_size, max_length, hidden_size))
        merged_mask = attention_mask.new_zeros((batch_size, max_length))
        merged_labels = None
        if labels is not None:
            labels = labels.to(device)
            merged_labels = labels.new_full((batch_size, max_length), IGNORE_TOKEN_ID)

        rows, columns = torch.where(text_valid & ~speech_tokens)
        destinations = ends[rows, columns]
        merged[rows, destinations] = inputs_embeds[rows, columns]
        merged_mask[rows, destinations] = 1
        if merged_labels is not None:
            merged_labels[rows, destinations] = labels[rows, columns]

        speech_starts = ends[speech_tokens] - speech_lens + 1
        valid_speech = (
            torch.arange(max_speech_len, device=device)[None, :] < speech_lens[:, None]
        )
        rows, frames = torch.where(valid_speech)
        destinations = speech_starts[rows] + frames
        merged[rows, destinations] = speech_features[rows, frames]
        merged_mask[rows, destinations] = 1

        position_ids = merged_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(~merged_mask.bool(), 0)
        return merged, merged_mask, merged_labels, position_ids

    def forward(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.LongTensor = None,
    ):
        encoder_outs, encoder_out_lens = self._encode_features(feature, feature_lens)
        speech_features = self.encoder_projector(encoder_outs)
        speech_lens = torch.div(
            encoder_out_lens,
            self.encoder_projector.downsample_rate,
            rounding_mode="floor",
        )
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            labels,
            position_ids,
        ) = self._merge_input_ids_with_speech_features(
            speech_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            labels,
            speech_lens=speech_lens,
        )
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc = compute_accuracy(
                preds.detach()[:, :-1],
                labels.detach()[:, 1:],
                ignore_label=IGNORE_TOKEN_ID,
            )
        return model_outputs, acc

    def decode(
        self,
        feature: torch.Tensor = None,
        feature_lens: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        encoder_outs, encoder_out_lens = self._encode_features(feature, feature_lens)
        speech_features = self.encoder_projector(encoder_outs)
        speech_lens = torch.div(
            encoder_out_lens,
            self.encoder_projector.downsample_rate,
            rounding_mode="floor",
        )
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        (
            inputs_embeds,
            attention_mask,
            _,
            _,
        ) = self._merge_input_ids_with_speech_features(
            speech_features,
            inputs_embeds,
            input_ids,
            attention_mask,
            speech_lens=speech_lens,
            padding_side="left",
        )
        # generate derives and advances position_ids from the packed attention mask.
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 1),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
        )


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.
    Copied from https://github.com/X-LANCE/SLAM-LLM/blob/main/src/slam_llm/utils/metric.py
    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()
