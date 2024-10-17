#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                  Zengwei Yao)
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

import warnings
from typing import Tuple

import torch
from scaling import (
    Balancer,
    BiasNorm,
    Dropout3,
    FloatLike,
    Optional,
    ScaledConv2d,
    ScaleGrad,
    ScheduledFloat,
    SwooshL,
    SwooshR,
    Whiten,
)
from torch import Tensor, nn


class ConvNeXt(torch.nn.Module):
    """
    The simplified ConvNeXt module interpretation based on https://arxiv.org/pdf/2206.14747.pdf.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        ConvNeXt initialization.

        Parameters
        ----------
        num_channels : int
            The number of input and output channels for ConvNeXt module.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.padding = 3
        hidden_channels = num_channels * 3

        self.depthwise_conv = torch.nn.Conv2d(
            num_channels,
            num_channels,
            7,
            groups=num_channels,
            padding=(0, self.padding),  # time, freq
            device=device,
        )

        self.activation = SwooshL()
        self.pointwise_conv1 = torch.nn.Conv2d(num_channels, hidden_channels, 1, device=device)
        self.pointwise_conv2 = torch.nn.Conv2d(hidden_channels, num_channels, 1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the ConvNeXt module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            An input float tensor of shape (1, num_channels, num_input_frames, num_freqs).

        Returns
        -------
        torch.Tensor[torch.float32]
            A output float tensor of the same shape as input,
            (1, num_channels, num_output_frames, num_freqs).
        """

        bypass = x[:, :, self.padding: x.size(2) - self.padding]

        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x

        return x


class Conv2dSubsampling(torch.nn.Module):
    """
    Convolutional 2D subsampling module. It performs the prior subsampling
    (four times subsampling along the frequency axis and two times - along the time axis),
    and low-level descriptor feature extraction from the log mel feature input before passing
    it to zipformer encoder.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer1_channels: int,
        layer2_channels: int,
        layer3_channels: int,
        right_context: int,
        device: torch.device,
    ) -> None:
        """
        Conv2dSubsampling initialization.

        Parameters
        ----------
        input_dim : int
            The number of input channels. Corresponds to the
            number of features in the input feature tensor.
        output_dim : int
            The number of output channels.
        layer1_channels : int
            The number of output channels in the first Conv2d layer.
        layer2_channels : int
            The number of output channels in the second Conv2d layer.
        layer3_channels : int
            The number of output channels in the third Conv2d layer.
        right_context: int
            The look-ahead right context that is used to update the left cache.
        device : torch.device
            The device used to store the layer weights. Should be
            either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if input_dim < 7:
            raise ValueError(
                'The input feature dimension of the Conv2dSubsampling layer, can not be less than '
                'seven, otherwise the frequency subsampling will result with an empty output. '
                f'Expected input_dim to be at least 7 but got {input_dim}.',
            )

        self.right_context = right_context

        # Assume batch size is 1 and the right padding is 10,
        # see the forward method on why the right padding is 10.
        self.right_pad = torch.full(
            (1, 10, input_dim), ZERO_LOG_MEL, dtype=torch.float32, device=device,
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
                device=device,
            ),
            SwooshR(),
            torch.nn.Conv2d(layer1_channels, layer2_channels, 3, stride=2, device=device),
            SwooshR(),
            torch.nn.Conv2d(layer2_channels, layer3_channels, 3, stride=(1, 2), device=device),
            SwooshR(),
        )

        self.convnext = ConvNeXt(layer3_channels, device=device)

        out_width = (((input_dim - 1) // 2) - 1) // 2
        self.out = torch.nn.Linear(out_width * layer3_channels, output_dim, device=device)
        self.out_norm = BiasNorm(output_dim, device=device)

    def forward(
        self, x: torch.Tensor, cached_left_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the Conv2dSubsampling module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            An input float tensor of shape (1, num_frames, input_dim). An input feature tensor.
        cached_left_pad : torch.Tensor[torch.float32]
            A left cache float tensor of shape (1, 10, input_dim). Left cache is required
            to preserve the "same" left padding to the output of the Conv2dSubsampling module.
            See the get_init_states() documentation to understand why we need exactly ten frames
            of left padding for the Conv2dSubsampling module.

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - The processing output of the Conv2dSubsampling module
              of shape (1, subsampled_num_frames, output_dim).
            - The udated left cache tensor of shape (1, 10, input_dim).
        """

        x = torch.cat((cached_left_pad, x), dim=1)
        new_cached_left_pad = x[
            :,
            x.size(1) - self.right_context - cached_left_pad.size(1):
            x.size(1) - self.right_context,
        ]

        # Now when we concatenated the left cache with the input, we need to perform the right
        # padding of the input in a way to preserve the "same" type of padding, so that the output
        # of the module has the same duration as input (taking 2 times subsampling into account).
        # There are two possible outcomes depending on whether the the number of input frames is
        # even or odd, but both scenarios can be covered by 10 frames right padding.

        #                 x          :     right padding
        #     | | | | | | | | | | | |:| | | | | | | | | |  input
        #       | | | | | | | | | | |:| | | | | | | | |    first Conv2d output from self.conv
        #         |   |   |   |   |  :|   |   |   |        second Conv2d output from self.conv
        #             |   |   |   |  :|   |   |            third Conv2d output from self.conv
        #                         |  :                     Conv2d output from
        #                            :                     self.convnext.depthwise_conv
        #                            :
        #               x            :     right padding
        #   | | | | | | | | | | | | |:| | | | | | | | | |  input
        #     | | | | | | | | | | | |:| | | | | | | | |    first Conv2d output from self.conv
        #       |   |   |   |   |   |:  |   |   |   |      second Conv2d output from self.conv
        #           |   |   |   |   |:  |   |   |          third Conv2d output from self.conv
        #                       |   |:                     Conv2d output from
        #                            :                     self.convnext.depthwise_conv
        #                            :

        x = torch.cat((x, self.right_pad), dim=1)

        # (1, T, input_dim) -> (1, 1, T, input_dim) i.e., (N, C, H, W)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.convnext(x)

        # Now x is of shape (1, output_dim, T', ((input_dim - 1) // 2 - 1) // 2)
        b, c, t, f = x.size()  # b is equal to 1
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        # Now x is of shape (T', output_dim * layer3_channels))
        x = self.out(x)
        # Now x is of shape (T', output_dim)
        x = self.out_norm(x)

        return x, new_cached_left_pad

def get_init_states(input_dim: int, device: torch.device) -> torch.Tensor:
    """
    Get initial states for Conv2dSubsampling module. The Conv2dSubsampling.conv consists of three
    consecutive Conv2d layers with the kernel size 3 and no padding, also the middle Conv2d
    has a stride 2, while the rest have the default stride 1. We want to pad the input from the
    left side with cached_left_pad in the "same" way, so when we pass it through
    the Conv2dSubsampling.conv and Conv2dSubsampling.convnext we end up with exactly zero padding
    frames from the left.

      cached_left_pad  :          x
    | | | | | | | | | |:| | | | | | | | | | |      input
      | | | | | | | | |:| | | | | | | | | | |      first Conv2d output from Conv2dSubsampling.conv
        |   |   |   |  :|   |   |   |   |   | ...  second Conv2d output from Conv2dSubsampling.conv
            |   |   |  :|   |   |   |   |   |      third Conv2d output from Conv2dSubsampling.conv
                       :|   |   |   |   |   |      Conv2d output from
                       :                           Conv2dSubsampling.convnext.depthwise_conv

    As we can see from the picture above, in order to preserve the "same"
    padding from the left side we need
    ((((pad - 1) - 1) // 2) - 1) - 3 = 0 --> pad = 10.

    Parameters
    ----------
    input_dim : int
        The number of input channels.
        Corresponds to the number of features in the input of the Conv2dSubsampling module.
    device : torch.device
        The device used to store the left cache tensor.
        Either torch.device("cpu") or torch.device("cuda").

    Returns
    -------
    torch.Tensor[torch.float32]
        A left cache float tensor. The output shape is (1, 10, input_dim).
    """

    pad = 10
    cached_left_pad = torch.full(
        (1, pad, input_dim), ZERO_LOG_MEL, dtype=torch.float32, device=device,
    )

    return cached_left_pad
