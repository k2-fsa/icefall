#!/usr/bin/env python3
# Copyright         2024   The Chinese University of HK   (Author: Zengrui Jin)
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


from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from modules.conv import NormConv1d, NormConv2d


def get_padding(kernel_size, dilation=1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period,
        kernel_size=5,
        stride=3,
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super(DiscriminatorP, self).__init__()

        self.period = period
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList(
            [
                NormConv2d(
                    1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)
                ),
                NormConv2d(
                    32,
                    32,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(5, 1), 0),
                ),
                NormConv2d(
                    32,
                    32,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(5, 1), 0),
                ),
                NormConv2d(
                    32,
                    32,
                    (kernel_size, 1),
                    (stride, 1),
                    padding=(get_padding(5, 1), 0),
                ),
                NormConv2d(32, 32, (kernel_size, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = NormConv2d(32, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(
        self,
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super(DiscriminatorS, self).__init__()
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList(
            [
                NormConv1d(1, 32, 15, 1, padding=7),
                NormConv1d(32, 32, 41, 2, groups=4, padding=20),
                NormConv1d(32, 32, 41, 2, groups=16, padding=20),
                NormConv1d(32, 32, 41, 4, groups=16, padding=20),
                NormConv1d(32, 32, 41, 4, groups=16, padding=20),
                NormConv1d(32, 32, 41, 1, groups=16, padding=20),
                NormConv1d(32, 32, 5, 1, padding=2),
            ]
        )
        self.conv_post = NormConv1d(32, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(
        self,
        n_filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: Tuple[int, int] = (3, 9),
        dilations: List[int] = [1, 2, 4],
        stride: Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = n_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap
