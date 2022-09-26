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


class Joiner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, C).
          decoder_out:
            Output from the decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        logit = encoder_out + decoder_out
        logit = torch.tanh(logit)

        output = self.output_linear(logit)

        return output

    def forward_lm(
        self, decoder_out: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Args:
          decoder_out:
            Output from the decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """

        # decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        logit = torch.tanh(decoder_out)
        # if debug:
        #     print(self.output_linear._parameters["weight"][0, :20])
        #     print(self.output_linear._parameters["weight"][10, :20])
        output = self.output_linear(logit)

        # check output every same row
        # remove blank
        # mask ?

        return output[:, :, 1:].log_softmax(dim=-1)

    def forward_lm_raw(
        self, decoder_out: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Args:
          decoder_out:
            Output from the decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """

        # decoder_out = decoder_out.unsqueeze(1)
        # Now decoder_out is (N, 1, U, C)

        logit = torch.tanh(decoder_out)
        # if debug:
        #     print(self.output_linear._parameters["weight"][0, :20])
        #     print(self.output_linear._parameters["weight"][10, :20])
        output = self.output_linear(logit)

        # check output every same row
        # remove blank
        # mask ?

        return output[:, :, 1:],  output[:, :, :]
