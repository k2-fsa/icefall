# Copyright      2022  Xiaomi Corporation     (Author: Zengwei Yao)
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
#
# It is modified based on https://github.com/pytorch/audio/blob/main/torchaudio/models/emformer.py.  # noqa

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledConv2d,
    ScaledLinear,
)

from icefall.utils import make_pad_mask


class RelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module.

    See : Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py  # noqa

    Suppose:
      i -> position of query,
      j -> position of key(value),
    we use positive relative position embedding when key(value) is to the
    left of query(i.e., i > j) and negative embedding otherwise.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(
        self, d_model: int, dropout: float, max_len: int = 5000
    ) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.pos_len = max_len
        self.neg_len = max_len
        self.gen_pe_positive()
        self.gen_pe_negative()

    def gen_pe_positive(self) -> None:
        """Generate the positive positional encodings."""
        pe_positive = torch.zeros(self.pos_len, self.d_model)
        position_positive = torch.arange(
            0, self.pos_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position_positive * div_term)
        pe_positive[:, 1::2] = torch.cos(position_positive * div_term)
        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"  # noqa
        self.pe_positive = torch.flip(pe_positive, [0])

    def gen_pe_negative(self) -> None:
        """Generate the negative positional encodings."""
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use positive relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_negative = torch.zeros(self.neg_len, self.d_model)
        position_negative = torch.arange(
            0, self.neg_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_negative[:, 0::2] = torch.sin(-1 * position_negative * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position_negative * div_term)
        self.pe_negative = pe_negative

    def get_pe(
        self,
        pos_len: int,
        neg_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Get positional encoding given positive length and negative length."""
        if self.pe_positive.dtype != dtype or str(
            self.pe_positive.device
        ) != str(device):
            self.pe_positive = self.pe_positive.to(dtype=dtype, device=device)
        if self.pe_negative.dtype != dtype or str(
            self.pe_negative.device
        ) != str(device):
            self.pe_negative = self.pe_negative.to(dtype=dtype, device=device)
        pe = torch.cat(
            [
                self.pe_positive[self.pos_len - pos_len :],
                self.pe_negative[1:neg_len],
            ],
            dim=0,
        )
        return pe

    def forward(
        self,
        x: torch.Tensor,
        pos_len: int,
        neg_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale input x and get positional encoding.
        Args:
            x (torch.Tensor): Input tensor (`*`).

        Returns:
          torch.Tensor:
            Encoded tensor of shape (`*`).
          torch.Tensor:
            Position embedding of shape (pos_len + neg_len - 1, `*`).
        """
        x = x * self.xscale
        if pos_len > self.pos_len:
            self.pos_len = pos_len
            self.gen_pe_positive()
        if neg_len > self.neg_len:
            self.neg_len = neg_len
            self.gen_pe_negative()
        pos_emb = self.get_pe(pos_len, neg_len, x.device, x.dtype)
        return self.dropout(x), self.dropout(pos_emb)
