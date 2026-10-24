# Copyright    2024  (authors: Ruizhe Huang)
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

from typing import List, Optional

import torch
from context_encoder import ContextEncoder


def last_hidden_states(hn: torch.Tensor) -> torch.Tensor:
    """Concatenate the final forward and backward hidden states of the last
    layer of a bidirectional LSTM; hn has shape
    (num_layers * 2, batch_size, hidden_size)."""
    return torch.cat((hn[-2], hn[-1]), dim=1)


class ContextEncoderLSTM(ContextEncoder):
    """Embeds each biasing word from its BPE tokens with a (bidirectional)
    LSTM."""

    def __init__(
        self,
        vocab_size: int,
        context_encoder_dim: int,
        output_dim: int,
        num_layers: int,
        num_directions: int,
    ):
        super().__init__()
        assert num_directions == 2, "Only bidirectional LSTMs are supported"
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.context_encoder_dim = context_encoder_dim

        self.embed = torch.nn.Embedding(vocab_size, context_encoder_dim)
        self.rnn = torch.nn.LSTM(
            input_size=context_encoder_dim,
            hidden_size=context_encoder_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = torch.nn.Linear(
            context_encoder_dim * self.num_directions, output_dim
        )

    def forward(
        self,
        word_list: torch.Tensor,
        word_lengths: List[int],
        is_encoder_side: Optional[bool] = None,
    ) -> torch.Tensor:
        out = self.embed(word_list)
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, batch_first=True, lengths=word_lengths, enforce_sorted=False
        )
        _, (hn, _) = self.rnn(out)
        return self.linear(last_hidden_states(hn))
