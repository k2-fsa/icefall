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


# TODO(fangjun): Support switching between LSTM and GRU
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_id: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          embedding_dim:
            Dimension of the input embedding.
          blank_id:
            The ID of the blank symbol.
          num_layers:
            Number of LSTM layers.
          hidden_dim:
            Hidden dimension of LSTM layers.
          output_dim:
            Output dimension of the decoder.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        # TODO(fangjun): Use layer normalized LSTM
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.blank_id = blank_id
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U) with BOS prepended.
          states:
            A tuple of two tensors containing the states information of
            LSTM layers in this decoder.
        Returns:
          Return a tuple containing:

            - rnn_output, a tensor of shape (N, U, C)
            - (h, c), containing the state information for LSTM layers.
              Both are of shape (num_layers, N, C)
        """
        embedding_out = self.embedding(y)
        embedding_out = self.embedding_dropout(embedding_out)
        rnn_out, (h, c) = self.rnn(embedding_out, states)
        out = self.output_linear(rnn_out)

        return out, (h, c)
