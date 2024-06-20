# Copyright    2023  Xiaomi Corp.        (authors: Zengwei Yao)
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


from typing import List, Tuple

import torch
import torch.nn as nn
from encoder_interface import EncoderInterface


class CTCAttentionModel(nn.Module):
    """Hybrid CTC & Attention decoder model."""

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        encoder_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the Zipformer encoder model. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the attention decoder.
          encoder_dim:
            The embedding dimension of encoder.
          vocab_size:
            The vocabulary size.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder = encoder
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1),
        )
        # Attention decoder
        self.decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        token_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          token_ids:
            A list of token id list.

        Returns:
          - ctc_output, ctc log-probs
          - att_loss, attention decoder loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert x.size(0) == x_lens.size(0) == len(token_ids)

        # encoder forward
        encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        # compute ctc log-probs
        ctc_output = self.ctc_output(encoder_out)

        # compute attention decoder loss
        att_loss = self.decoder.calc_att_loss(encoder_out, x_lens, token_ids)

        return ctc_output, att_loss
