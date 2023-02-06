# Copyright      2022  Xiaomi Corp.        (authors: Yifan Yang)
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from scaling import (
    ActivationBalancer,
    ScaledConv1d,
)


class LConv(nn.Module):
    """A convolution module to prevent information loss."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        bias: bool = True,
    ):
        """
        Args:
          channels:
            Dimension of the input embedding, and of the lconv output.
        """
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        self.deriv_balancer1 = ActivationBalancer(
            2 * channels,
            channel_dim=1,
            max_abs=10.0,
            min_positive=0.05,
            max_positive=1.0,
        )

        self.depthwise_conv = nn.Conv1d(
            2 * channels,
            2 * channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=2 * channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            2 * channels,
            channel_dim=1,
            min_positive=0.05,
            max_positive=1.0,
            max_abs=20.0,
        )

        self.pointwise_conv2 = ScaledConv1d(
            2 * channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: A 3-D tensor of shape (N, T, C).
        Returns:
            Return a tensor of shape (N, T, C).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(0, 2, 1)  # (#batch, channels, time).

        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)

        x = self.pointwise_conv2(x)  # (batch, channels, time)

        return x.permute(0, 2, 1)
