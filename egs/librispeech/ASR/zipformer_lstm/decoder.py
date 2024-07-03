# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengrui Jin,
#                                                       Yifan Yang,)
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
import torch.nn as nn
import torch.nn.functional as F
from scaling import Balancer


class Decoder(nn.Module):
    """LSTM decoder."""

    def __init__(
        self,
        vocab_size: int,
        blank_id: int,
        decoder_dim: int,
        num_layers: int,
        hidden_dim: int,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          blank_id:
            The ID of the blank symbol.
          decoder_dim:
            Dimension of the input embedding.
          num_layers:
            Number of LSTM layers.
          hidden_dim:
            Hidden dimension of LSTM layers.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            decoder_dim=decoder_dim,
        )
        # the balancers are to avoid any drift in the magnitude of the
        # embeddings, which would interact badly with parameter averaging.
        self.balancer = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

        self.blank_id = blank_id

        self.vocab_size = vocab_size

        # self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.rnn = nn.LSTM(
            input_size=decoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
  
        self.balancer2 = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        embedding_out = self.balancer(embedding_out)

        rnn_out, (h, c) = self.rnn(embedding_out, states)

        rnn_out = F.relu(rnn_out)
        rnn_out = self.balancer2(rnn_out)

        return rnn_out, (h, c)
