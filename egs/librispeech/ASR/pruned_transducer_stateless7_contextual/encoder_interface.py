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

from typing import Tuple

import torch
import torch.nn as nn


class EncoderInterface(nn.Module):
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (batch_size, input_seq_len, num_features)
            containing the input features.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames
            in `x` before padding.
        Returns:
          Return a tuple containing two tensors:
            - encoder_out, a tensor of (batch_size, out_seq_len, output_dim)
              containing unnormalized probabilities, i.e., the output of a
              linear layer.
            - encoder_out_lens, a tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        raise NotImplementedError("Please implement it in a subclass")
