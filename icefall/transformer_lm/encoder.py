# Copyright (c)  2021  Xiaomi Corporation (authors: Xiaoyu Yang)
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

import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from icefall.transformer_lm.attention import RelPositionMultiheadAttention
from icefall.transformer_lm.scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)
from icefall.utils import is_jit_tracing, make_pad_mask


class Transformer(torch.nn.Module):
    """_summary_

    Args:
        input_dim (int): Input feature dimension
        d_mode (int): The dimension of the transformer
        dim_feedforward (int ): The dimension of the ffw module
        nhead (int): The number of attention heads
        dropout_rate (float): dropout rate
        att_dropout (float): dropout rate in attention module
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dim_feedforward: int,
        nhead: int = 4,
        num_layers: int = 6,
        dropout_rate: float = 0.1,
        att_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder_layers = num_layers
        self.d_model = d_model

        self.embed = ScaledLinear(input_dim, d_model)
        self.norm_before = BasicNorm(d_model, learn_eps=False)

        self.encoder_pos = RelPositionalEncoding(d_model, dropout_rate)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout_rate=dropout_rate,
        )

        self.encoder = TransformerEncoder(encoder_layer, num_layers)

    def _create_attention_mask(self, x_lens: torch.Tensor):
        # create a 2D attention mask to mask out
        # the upper right half of the attention matrix
        max_len = max(x_lens)
        ones = torch.ones(max_len, max_len, device=x_lens.device, dtype=torch.bool)
        return torch.triu(ones, diagonal=1)

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transformer forward

        Args:
            x (torch.Tensor): Input tensor (B,T,input_dim)
            x_lens (torch.Tensor): The length of input tensors before padding (B,)

        Returns:
            Return a tuple of 2 tensors:
            - x: output feature of the transformer (B,T,d_model)
            - x_lens: output feature lens of the transformer
        """

        attention_mask = self._create_attention_mask(x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)

        x = self.norm_before(self.embed(x))

        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)

        x = self.encoder(
            x,
            pos_emb,
            mask=attention_mask,  # pass the attention mast
            src_key_padding_mask=src_key_padding_mask,
        )  # (T, N, C)

        x = x.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        return x, x_lens


class TransformerEncoder(torch.nn.Module):
    def __init__(self, encoder_layer: torch.nn.Module, num_layers: int) -> None:
        """TransformerEncoder is a stack of N encoder layers

        Args:
            encoder_layer (torch.nn.Module): an instance of the TransformerEncoderLayer()
            num_layers (int): Number of layers to be stacked
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """_summary_

        Args:
            src: the sequence to the encoder (required).
            pos_emb: Positional embedding tensor (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Returns:
            output: transformer encoded features
        """
        output = src

        for layer_index, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                src_key_padding_mask=src_key_padding_mask,
                src_mask=mask,
            )

        return output


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        dropout_rate: float,
    ):
        """TransformerEncoderLayer is made up of self-attn and feedforward module

        Args:
            d_model (int): The model size
            dim_feedforward (int): Dimension of ffw module
            nhead (int): Number of heads
            dropout_rate (float): Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)
        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout_rate),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.norm_final = BasicNorm(d_model)

        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        cache=None,
    ):
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            pos_emb: Positional embedding tensor (required).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            src_mask: the mask for the src sequence (optional).
        """
        src_orig = src

        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

        src = src + self.dropout(src_att)

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        return src


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        if is_jit_tracing():
            # 10k frames correspond to ~100k ms, e.g., 100 seconds, i.e.,
            # It assumes that the maximum input won't have more than
            # 10k frames.
            #
            # TODO(fangjun): Use torch.jit.script() for this module
            max_len = 10000

        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor, left_context: int = 0) -> None:
        """Reset the positional encodings."""
        x_size_1 = x.size(1) + left_context
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x_size_1 * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x_size_1, self.d_model)
        pe_negative = torch.zeros(x_size_1, self.d_model)
        position = torch.arange(0, x_size_1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        left_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            left_context (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        self.extend_pe(x, left_context)
        x_size_1 = x.size(1) + left_context
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x_size_1
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)
