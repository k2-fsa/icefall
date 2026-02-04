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
from typing import Tuple, Optional

import torch
from scaling import (
    ScaleLimiter,
    ScaledLinear,
    GaussNorm,
    FloatLike,
    get_max_similarity,
    ScaledConv2d,
    ScaleGrad,
    ScheduledFloat,
    SwashL,
    SwashR,
    CosineSimilarityLoss,
    with_loss,
)
from torch import Tensor, nn


class AddNoise(nn.Module):
    # assume Conv2d-style input: (N, C, H, W)
    def __init__(self, rel_noise_scale: float):
        super().__init__()
        self.rel_noise_scale = rel_noise_scale

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        eps = 3.0e-08
        noise_scale = ((x ** 2).mean(dim=(1,2,3), keepdim=True) + eps).sqrt() * self.rel_noise_scale
        return x + noise_scale * torch.randn_like(x)



class ConvNeXt(nn.Module):
    """
    Our interpretation of the ConvNeXt module as used in https://arxiv.org/pdf/2206.14747.pdf
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 3,
        kernel_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        hidden_channels = channels * hidden_ratio

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )

        self.pointwise_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=hidden_channels, kernel_size=1,
        )

        self.activation = SwashL()

        self.pointwise_conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
        )


    def forward(
        self, x: Tensor,
    ) -> Tensor:
        """
        x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)

        The returned value has the same shape as x.
        """
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cached_left_pad: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)
            cached_left_pad: (batch_size, num_channels, left_pad, num_freqs)

        Returns:
            - The returned value has the same shape as x.
            - Updated cached_left_pad.
        """
        padding = self.padding

        # The length without right padding for depth-wise conv
        T = x.size(2) - padding[0]

        bypass = x[:, :, :T, :]

        # Pad left side
        assert cached_left_pad.size(2) == padding[0], (
            cached_left_pad.size(2),
            padding[0],
        )
        x = torch.cat([cached_left_pad, x], dim=2)
        # Update cached left padding
        cached_left_pad = x[:, :, T : padding[0] + T, :]

        # depthwise_conv
        x = torch.nn.functional.conv2d(
            x,
            weight=self.depthwise_conv.weight,
            bias=self.depthwise_conv.bias,
            padding=(0, padding[1]),
            groups=self.depthwise_conv.groups,
        )
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x
        return x, cached_left_pad


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = (T-3)//2 - 2 == (T-7)//2

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 16,
        layer2_channels: int = 64,
        layer3_channels: int = 128,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >=7, in_channels >=7
          out_channels
            Output dim. The output shape is (N, (T-3)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
          bottleneck:
            bottleneck dimension for 1d squeeze-excite
        """
        assert in_channels >= 7
        self.in_channels = in_channels
        super().__init__()

        # The ScaleGrad module is there to prevent the gradients
        # w.r.t. the weight or bias of the first Conv2d module in self.conv from
        # exceeding the range of fp16 when using automatic mixed precision (amp)
        # training.  (The second one is necessary to stop its bias from getting
        # a too-large gradient).

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            AddNoise(rel_noise_scale=5.0e-03),
            SwashR(),
            nn.Conv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            SwashR(),
            nn.Conv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=(1, 2),  # (time, freq)
            ),
            SwashR(),
        )


        # just one convnext layer
        self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7))

        # (in_channels-3)//4
        self.out_width = (((in_channels - 1) // 2) - 1) // 2
        self.layer3_channels = layer3_channels


        # scale it up a bit, else the output is quite small.
        self.out = ScaledLinear(self.out_width * layer3_channels, out_channels)

        self.out_norm = GaussNorm()

    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor, aux_loss_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in

        Returns:
          - a tensor of shape (N, (T-7)//2, odim)
          - output lengths, of shape (batch_size,)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        x = self.convnext(x)

        # Now x is of shape (N, odim, (T-7)//2, (idim-3)//4)
        b, c, t, f = x.size()

        x = x.transpose(1, 2).reshape(b, t, c * f)
        # now x: (N, (T-7)//2, out_width * layer3_channels))

        x = 0.1 * self.out(x)
        # Now x is of shape (N, (T-7)//2, odim)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            x_lens = (x_lens - 7) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_lens = (x_lens - 7) // 2

        key_padding_mask = torch.arange(0, x.shape[1], device=x.device) >= x_lens.unsqueeze(-1)
        # key_padding_mask: (N, (T-7)//2)
        x = x.permute(1, 0, 2)
        # x: (time, batch, channels)
        x = self.out_norm(x)
        x = x.permute(1, 0, 2)
        # x: (batch, time, channels)

        assert x.size(1) == x_lens.max().item(), (x.size(1), x_lens.max())

        return x, x_lens

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        cached_left_pad: Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in

        Returns:
          - a tensor of shape (N, (T-7)//2, odim)
          - output lengths, of shape (batch_size,)
          - updated cache
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)

        # T' = (T-7)//2
        x = self.conv(x)

        # T' = (T-7)//2-3
        x, cached_left_pad = self.convnext.streaming_forward(
            x, cached_left_pad=cached_left_pad
        )

        # Now x is of shape (N, odim, T', ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()

        x = x.transpose(1, 2).reshape(b, t, c * f)
        # now x: (N, T', out_width * layer3_channels))

        x = 0.1 * self.out(x)
        # Now x is of shape (N, T', odim)
        x = self.out_norm(x)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert self.convnext.padding[0] == 3
            # The ConvNeXt module needs 3 frames of right padding after subsampling
            x_lens = (x_lens - 7) // 2 - 3
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # The ConvNeXt module needs 3 frames of right padding after subsampling
                assert self.convnext.padding[0] == 3
                x_lens = (x_lens - 7) // 2 - 3

        assert x.size(1) == x_lens.max().item(), (x.shape, x_lens.max())

        return x, x_lens, cached_left_pad

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Get initial states for Conv2dSubsampling module.
        It is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        """
        left_pad = self.convnext.padding[0]
        freq = self.out_width
        channels = self.layer3_channels
        cached_embed_left_pad = torch.zeros(batch_size, channels, left_pad, freq).to(
            device
        )

        return cached_embed_left_pad
