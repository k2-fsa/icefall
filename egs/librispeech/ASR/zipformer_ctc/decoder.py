# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from label_smoothing import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_sequence
from transformer import PositionalEncoding, TransformerDecoderLayer


class Decoder(nn.Module):
    """This class implements Transformer based decoder for an attention-based encoder-decoder
    model.
    """

    def __init__(
        self,
        num_layers: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        """
        Args:
          num_layers:
            Number of layers.
          num_classes:
            Number of tokens of the modeling unit including blank.
          d_model:
            Dimension of the input embedding, and of the decoder output.
        """
        super().__init__()

        if num_layers > 0:
            self.decoder_num_class = num_classes  # bpe model already has sos/eos symbol

            self.decoder_embed = nn.Embedding(
                num_embeddings=self.decoder_num_class, embedding_dim=d_model
            )
            self.decoder_pos = PositionalEncoding(d_model, dropout)

            decoder_layer = TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                normalize_before=normalize_before,
            )

            if normalize_before:
                decoder_norm = nn.LayerNorm(d_model)
            else:
                decoder_norm = None

            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_layers,
                norm=decoder_norm,
            )

            self.decoder_output_layer = torch.nn.Linear(d_model, self.decoder_num_class)
            self.decoder_criterion = LabelSmoothingLoss()
        else:
            self.decoder_criterion = None

    @torch.jit.export
    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[List[int]],
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder with shape (T, N, C)
          memory_key_padding_mask:
            The padding mask from the encoder.
          token_ids:
            A list-of-list IDs. Each sublist contains IDs for an utterance.
            The IDs can be either phone IDs or word piece IDs.
          sos_id:
            sos token id
          eos_id:
            eos token id
        Returns:
            A scalar, the **sum** of label smoothing loss over utterances
            in the batch without any normalization.
        """
        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=float(eos_id))

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(ys_out, batch_first=True, padding_value=float(-1))

        device = memory.device
        ys_in_pad = ys_in_pad.to(device)
        ys_out_pad = ys_out_pad.to(device)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(device)

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (N, T) -> (N, T, C)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T, N, C)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        pred_pad = self.decoder_output_layer(pred_pad)  # (N, T, C)

        decoder_loss = self.decoder_criterion(pred_pad, ys_out_pad)

        return decoder_loss

    @torch.jit.export
    def decoder_nll(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        token_ids: List[torch.Tensor],
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """
        Args:
          memory:
            It's the output of the encoder with shape (T, N, C)
          memory_key_padding_mask:
            The padding mask from the encoder.
          token_ids:
            A list-of-list IDs (e.g., word piece IDs).
            Each sublist represents an utterance.
          sos_id:
            The token ID for SOS.
          eos_id:
            The token ID for EOS.
        Returns:
            A 2-D tensor of shape (len(token_ids), max_token_length)
            representing the cross entropy loss (i.e., negative log-likelihood).
        """
        # The common part between this function and decoder_forward could be
        # extracted as a separate function.
        if isinstance(token_ids[0], torch.Tensor):
            # This branch is executed by torchscript in C++.
            # See https://github.com/k2-fsa/k2/pull/870
            # https://github.com/k2-fsa/k2/blob/3c1c18400060415b141ccea0115fd4bf0ad6234e/k2/torch/bin/attention_rescore.cu#L286
            token_ids = [tolist(t) for t in token_ids]

        ys_in = add_sos(token_ids, sos_id=sos_id)
        ys_in = [torch.tensor(y) for y in ys_in]
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=float(eos_id))

        ys_out = add_eos(token_ids, eos_id=eos_id)
        ys_out = [torch.tensor(y) for y in ys_out]
        ys_out_pad = pad_sequence(ys_out, batch_first=True, padding_value=float(-1))

        device = memory.device
        ys_in_pad = ys_in_pad.to(device, dtype=torch.int64)
        ys_out_pad = ys_out_pad.to(device, dtype=torch.int64)

        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(device)

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=eos_id)
        # TODO: Use length information to create the decoder padding mask
        # We set the first column to False since the first column in ys_in_pad
        # contains sos_id, which is the same as eos_id in our current setting.
        tgt_key_padding_mask[:, 0] = False

        tgt = self.decoder_embed(ys_in_pad)  # (B, T) -> (B, T, F)
        tgt = self.decoder_pos(tgt)
        tgt = tgt.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)
        pred_pad = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (T, B, F)
        pred_pad = pred_pad.permute(1, 0, 2)  # (T, B, F) -> (B, T, F)
        pred_pad = self.decoder_output_layer(pred_pad)  # (B, T, F)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            pred_pad.view(-1, self.decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=-1,
            reduction="none",
        )

        nll = nll.view(pred_pad.shape[0], -1)

        return nll


def add_sos(token_ids: List[List[int]], sos_id: int) -> List[List[int]]:
    """Prepend sos_id to each utterance.
    Args:
      token_ids:
        A list-of-list of token IDs. Each sublist contains
        token IDs (e.g., word piece IDs) of an utterance.
      sos_id:
        The ID of the SOS token.
    Return:
      Return a new list-of-list, where each sublist starts
      with SOS ID.
    """
    return [[sos_id] + utt for utt in token_ids]


def add_eos(token_ids: List[List[int]], eos_id: int) -> List[List[int]]:
    """Append eos_id to each utterance.
    Args:
      token_ids:
        A list-of-list of token IDs. Each sublist contains
        token IDs (e.g., word piece IDs) of an utterance.
      eos_id:
        The ID of the EOS token.
    Return:
      Return a new list-of-list, where each sublist ends
      with EOS ID.
    """
    return [utt + [eos_id] for utt in token_ids]


def decoder_padding_mask(ys_pad: torch.Tensor, ignore_id: int = -1) -> torch.Tensor:
    """Generate a length mask for input.
    The masked position are filled with True,
    Unmasked positions are filled with False.
    Args:
      ys_pad:
        padded tensor of dimension (batch_size, input_length).
      ignore_id:
        the ignored number (the padding number) in ys_pad
    Returns:
      Tensor:
        a bool tensor of the same shape as the input tensor.
    """
    ys_mask = ys_pad == ignore_id
    return ys_mask


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a square mask for the sequence. The masked positions are
    filled with float('-inf'). Unmasked positions are filled with float(0.0).
    The mask can be used for masked self-attention.
    For instance, if sz is 3, it returns::
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0]])
    Args:
      sz: mask size
    Returns:
      A square mask of dimension (sz, sz)
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def tolist(t: torch.Tensor) -> List[int]:
    """Used by jit"""
    return torch.jit.annotate(List[int], t.tolist())
