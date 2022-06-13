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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Joiner(nn.Module):
    def __init__(self, input_dim: int, inner_dim: int, output_dim: int):
        super().__init__()

        self.inner_linear = nn.Linear(input_dim, inner_dim)
        self.output_linear = nn.Linear(inner_dim, output_dim)

    def forward(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C) during
            training or (N, C) in case of streaming decoding.
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C) during
            training or (N, C) in case of streaming decoding.
          Return a tensor of shape (N, T, s_range, C).
        """
        assert encoder_out.ndim == decoder_out.ndim
        assert encoder_out.ndim in (2, 4)
        assert encoder_out.shape == decoder_out.shape

        logit = encoder_out + decoder_out

        logit = self.inner_linear(torch.tanh(logit))

        output = self.output_linear(F.relu(logit))

        return output
