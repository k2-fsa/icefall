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
from context_encoder_lstm import last_hidden_states


class ContextEncoderReused(ContextEncoder):
    """Embeds each biasing word by running the (frozen) transducer decoder
    over its BPE tokens, followed by a bidirectional LSTM."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        decoder_dim: int,
        output_dim: int,
        num_lstm_layers: int,
        num_lstm_directions: int,
    ):
        super().__init__()
        assert num_lstm_directions == 2, "Only bidirectional LSTMs are supported"

        hidden_size = output_dim * 2
        self.rnn = torch.nn.LSTM(
            input_size=decoder_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_lstm_layers > 1 else 0,
        )
        self.linear = torch.nn.Linear(hidden_size * num_lstm_directions, output_dim)
        self.decoder = decoder

    def forward(
        self,
        word_list: torch.Tensor,
        word_lengths: List[int],
        is_encoder_side: Optional[bool] = None,
    ) -> torch.Tensor:
        sos_id = self.decoder.blank_id
        sos_list = torch.full(
            (word_list.size(0), 1), sos_id, device=word_list.device
        )
        sos_word_list = torch.cat((sos_list, word_list), dim=1)
        word_lengths = [x + 1 for x in word_lengths]

        # (num_words, max_word_len + 1, decoder_dim)
        out = self.decoder(sos_word_list)
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, batch_first=True, lengths=word_lengths, enforce_sorted=False
        )
        _, (hn, _) = self.rnn(out)
        return self.linear(last_hidden_states(hn))
