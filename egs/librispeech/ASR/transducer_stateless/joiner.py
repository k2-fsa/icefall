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


class Joiner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        encoder_out_len: torch.Tensor,
        decoder_out_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, self.input_dim).
          decoder_out:
            Output from the decoder. Its shape is (N, U, self.input_dim).
          encoder_out_len:
            A 1-D tensor of shape (N,) containing valid number of frames
            before padding in `encoder_out`.
          decoder_out_len:
            A 1-D tensor of shape (N,) containing valid number of frames
            before padding in `decoder_out`.
        Returns:
          Return a tensor of shape (sum_all_TU, self.output_dim).
        """
        assert encoder_out.ndim == decoder_out.ndim == 3
        assert encoder_out.size(0) == decoder_out.size(0)
        assert encoder_out.size(2) == self.input_dim
        assert decoder_out.size(2) == self.input_dim

        N = encoder_out.size(0)

        encoder_out_len: List[int] = encoder_out_len.tolist()
        decoder_out_len: List[int] = decoder_out_len.tolist()

        encoder_out_list = [encoder_out[i, : encoder_out_len[i], :] for i in range(N)]

        decoder_out_list = [decoder_out[i, : decoder_out_len[i], :] for i in range(N)]

        x = [
            e.unsqueeze(1) + d.unsqueeze(0)
            for e, d in zip(encoder_out_list, decoder_out_list)
        ]

        x = [p.reshape(-1, self.input_dim) for p in x]
        x = torch.cat(x)

        activations = torch.tanh(x)

        logits = self.output_linear(activations)

        return logits
