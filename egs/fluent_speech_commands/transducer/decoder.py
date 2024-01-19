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

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_id: int,
        num_layers: int,
        hidden_dim: int,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit.
          embedding_dim:
            Dimension of the input embedding.
          blank_id:
            The ID of the blank symbol.
          num_layers:
            Number of RNN layers.
          hidden_dim:
            Hidden dimension of RNN layers.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for RNN layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.blank_id = blank_id
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          states:
            A tuple of two tensors containing the states information of
            RNN layers in this decoder.
        Returns:
          Return a tuple containing:

            - rnn_output, a tensor of shape (N, U, C)
            - (h, c), which contain the state information for RNN layers.
              Both are of shape (num_layers, N, C)
        """
        embedding_out = self.embedding(y)
        embedding_out = self.embedding_dropout(embedding_out)
        rnn_out, (h, c) = self.rnn(embedding_out, states)
        out = self.output_linear(rnn_out)

        return out, (h, c)
