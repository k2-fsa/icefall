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


# We use a TDNN model as encoder, as it works very well with CTC training
# for this tiny dataset.
class Tdnn(nn.Module):
    def __init__(self, num_features: int, output_dim: int):
        """
        Args:
          num_features:
            Model input dimension.
          ouput_dim:
            Model output dimension
        """
        super().__init__()

        # Note: We don't use paddings inside conv layers
        self.tdnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=32,
                kernel_size=3,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                dilation=4,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=32, affine=False),
        )
        self.output_linear = nn.Linear(in_features=32, out_features=output_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            The input tensor with shape (N, T, C)
          x_lens:
            It contains the number of frames in each utterance in x
            before padding.

        Returns:
          Return a tuple with 2 tensors:

            - logits, a tensor of shape (N, T, C)
            - logit_lens, a tensor of shape (N,)
        """
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        x = self.tdnn(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        logits = self.output_linear(x)

        # the first conv layer reduces T by 3-1 frames
        # the second layer reduces T by (5-1)*2 frames
        # the second layer reduces T by (5-1)*4 frames
        # Number of output frames is 2 + 4*2 + 4*4 = 2 + 8 + 16 = 26
        x_lens = x_lens - 26
        return logits, x_lens
