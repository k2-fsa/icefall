#!/usr/bin/env python3
# Copyright (c)  2022 Spacetouch Inc. (author: Tiance Wang)
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
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from scaling import ActivationBalancer, DoubleSwish
from torch import Tensor, nn


class Conv1dNet(EncoderInterface):
    """
    1D Convolution network with causal squeeze and excitation
    module and optional skip connections.

    Latency: 80ms + (conv_layers+1) // 2 * 40ms, assuming 10ms stride.

    Args:
        output_dim (int): Number of output channels of the last layer.
        input_dim (int): Number of input features
        conv_layers (int): Number of convolution layers,
        excluding the subsampling layers.
        channels (int): Number of output channels for each layer,
        except the last layer.
        subsampling_factor (int): The subsampling factor for the model.
        skip_add (bool): Whether to use skip connection for each convolution layer.
        dscnn (bool): Whether to use depthwise-separated convolution.
        activation (str): Activation function type.
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = 80,
        conv_layers: int = 10,
        channels: int = 256,
        subsampling_factor: int = 4,
        skip_add: bool = False,
        dscnn: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert subsampling_factor == 4, "Only support subsampling = 4"

        self.conv_layers = conv_layers
        self.skip_add = skip_add
        # 80ms latency for subsample_layer
        self.subsample_layer = nn.Sequential(
            conv1d_bn_block(
                input_dim, channels, 9, stride=2, activation=activation, dscnn=dscnn
            ),
            conv1d_bn_block(
                channels, channels, 5, stride=2, activation=activation, dscnn=dscnn
            ),
        )

        self.conv_blocks = nn.ModuleList()
        cin = [channels] * conv_layers
        cout = [channels] * (conv_layers - 1) + [output_dim]

        # Use causal and standard convolution alternatively
        for ly in range(conv_layers):
            self.conv_blocks.append(
                nn.Sequential(
                    conv1d_bn_block(
                        cin[ly],
                        cout[ly],
                        3,
                        activation=activation,
                        dscnn=dscnn,
                        causal=ly % 2,
                    ),
                    CausalSqueezeExcite1d(cout[ly], 16, 30),
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (batch_size, seq_len, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (batch_size, output_seq_len, encoder_dims)
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        x = self.subsample_layer(x)
        for idx, layer in enumerate(self.conv_blocks):
            if self.skip_add and 0 < idx < self.conv_layers - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        lengths = x_lens >> 2
        return x, lengths


def get_activation(
    name: str,
    channels: int,
    channel_dim: int = -1,
    min_val: int = 0,
    max_val: int = 1,
) -> nn.Module:
    """
    Get activation function from name in string.

    Args:
        name: activation function name
        channels: only used for PReLU, should be equal to x.shape[1].
        channel_dim: the axis/dimension corresponding to the channel,
            interprted as an offset from the input's ndim if negative.
            e.g. for NCHW tensor, channel_dim = 1
        min_val: minimum value of hardtanh
        max_val: maximum value of hardtanh

    Returns:
        The activation function module

    """
    act_layer = nn.Identity()
    name = name.lower()
    if name == "prelu":
        act_layer = nn.PReLU(channels)
    elif name == "relu":
        act_layer = nn.ReLU()
    elif name == "relu6":
        act_layer = nn.ReLU6()
    elif name == "hardtanh":
        act_layer = nn.Hardtanh(min_val, max_val)
    elif name in ["swish", "silu"]:
        act_layer = nn.SiLU()
    elif name == "elu":
        act_layer = nn.ELU()
    elif name == "doubleswish":
        act_layer = nn.Sequential(
            ActivationBalancer(num_channels=channels, channel_dim=channel_dim),
            DoubleSwish(),
        )
    elif name == "":
        act_layer = nn.Identity()
    else:
        raise Exception(f"Unknown activation function: {name}")

    return act_layer


class CausalSqueezeExcite1d(nn.Module):
    """
    Causal squeeze and excitation module with input and output shape
    (batch, channels, time). The global average pooling in the original
    SE module is replaced by a causal filter, so
    the layer does not introduce any algorithmic latency.

    Args:
        channels (int): Number of channels
        reduction (int): channel reduction rate
        context_window (int): Context window size for the moving average operation.
        For EMA, the smoothing factor is 1 / context_window.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        context_window: int = 10,
    ) -> None:
        super(CausalSqueezeExcite1d, self).__init__()

        assert channels >= reduction

        self.context_window = context_window
        c_squeeze = channels // reduction
        self.linear1 = nn.Linear(channels, c_squeeze, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(c_squeeze, channels, bias=True)
        self.act2 = nn.Sigmoid()

        # EMA worked better than MA empirically
        # self.avg_filter = self.moving_avg
        self.avg_filter = self.exponential_moving_avg
        self.ema_matrix = torch.tensor([0])
        self.ema_matrix_size = 0

    def _precompute_ema_matrix(self, N: int, device: torch.device):
        a = 1.0 / self.context_window  # smoothing factor
        w = [[(1 - a) ** k * a for k in range(n, n - N, -1)] for n in range(N)]
        w = torch.tensor(w).to(device).tril()
        w[:, 0] *= self.context_window
        self.ema_matrix = w.T
        self.ema_matrix_size = N

    def exponential_moving_avg(self, x: Tensor) -> Tensor:
        """
        Exponential moving average filter, which is calculated as:
            y[t] = (1-a) * y[t-1] + a * x[t]
        where a = 1 / self.context_window is the smoothing factor.

        For training, the iterative version is too slow. A better way is
        to expand y[t] as a function of x[0..t] only and use matrix-vector multiplication.
        The weight matrix can be precomputed if the smoothing factor is fixed.
        """
        if self.training:
            # use matrix version to speed up training
            N = x.shape[-1]
            if N > self.ema_matrix_size:
                self._precompute_ema_matrix(N, x.device)
            y = torch.matmul(x, self.ema_matrix[:N, :N])
        else:
            # use iterative version to save memory
            a = 1.0 / self.context_window
            y = torch.empty_like(x)
            y[:, :, 0] = x[:, :, 0]
            for t in range(1, y.shape[-1]):
                y[:, :, t] = (1 - a) * y[:, :, t - 1] + a * x[:, :, t]
        return y

    def moving_avg(self, x: Tensor) -> Tensor:
        """
        Simple moving average with context_window as window size.
        """
        y = torch.empty_like(x)
        k = min(x.shape[2], self.context_window)
        w = [[1 / n] * n + [0] * (k - n - 1) for n in range(1, k)]
        w = torch.tensor(w, device=x.device)
        y[:, :, : k - 1] = torch.matmul(x[:, :, : k - 1], w.T)
        y[:, :, k - 1 :] = F.avg_pool1d(x, k, 1)
        return y

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, "Input is not a 3D tensor!"
        y = self.exponential_moving_avg(x)
        y = y.permute(0, 2, 1)  # make channel last for squeeze op
        y = self.act1(self.linear1(y))
        y = self.act2(self.linear2(y))
        y = y.permute(0, 2, 1)  # back to original shape
        y = x * y
        return y


def conv1d_bn_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
    activation: str = "relu",
    dscnn: bool = False,
    causal: bool = False,
) -> nn.Sequential:
    """
    Conv1d - batchnorm - activation block.
    If kernel size is even, output length = input length + 1.
    Otherwise, output and input lengths are equal.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): kernel size
        stride (int): convolution stride
        dilation (int): convolution dilation rate
        dscnn (bool): Use depthwise separated convolution.
        causal (bool): Use causal convolution
        activation (str): Activation function type.

    """
    if dscnn:
        return nn.Sequential(
            CausalConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            )
            if causal
            else nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            get_activation(activation, in_channels),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            get_activation(activation, out_channels),
        )
    else:
        return nn.Sequential(
            CausalConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                bias=False,
            )
            if causal
            else nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size // 2) * dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            get_activation(activation, out_channels),
        )


class CausalConv1d(nn.Module):
    """
    Causal convolution with padding automatically chosen to match input/output length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(CausalConv1d, self).__init__()
        assert kernel_size > 2

        self.padding = dilation * (kernel_size - 1)
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            self.padding,
            dilation,
            groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)[:, :, : -self.padding // self.stride]
