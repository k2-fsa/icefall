# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey)
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


import logging
import math
import random
from typing import Optional, Tuple, Union

import k2
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler replacement for LayerNorm. The observation this is based on,
    is that Transformer-type networks, especially with pre-normalization, sometimes seem to set one
    of the channel dimensions to a large constant value (e.g. 50), which "defeats" the LayerNorm
    because the output magnitude is then not strongly dependent on the other (useful) channels.
    Presumably the weight and bias of the LayerNorm are required to allow it to do this. Instead,
    we give the BiasNorm a trainable bias that it can use when computing the scale for
    normalization. We also give it a scalar trainable scale on the output.
    """

    def __init__(self, num_channels: int, device: torch.device) -> None:
        """
        BiasNorm initialization.

        Parameters
        ----------
        num_channels : int
            The number of input channels.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        self.scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
        self.bias = torch.nn.Parameter(
            torch.zeros(num_channels, dtype=torch.float32, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass of the BiasNorm module.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len, num_channels). The module input.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of shape (1, seq_len, num_channels).
            A normalized output tensor of the same shape as input.
        """

        return x * self.scale / torch.mean((x - self.bias)**2, dim=2, keepdim=True)**0.5


class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    """
    Behaves like a depthwise 1D convolution, except that it is causal in a chunkwise way, as if we
    had a block-triangular attention mask.The chunk size is provided at test time, it should be kept
    in sync with the attention mask.

    This has a little more than twice the parameters of a conventional depthwise conv1d module:
    we implement it by having one depthwise convolution, of half the width, that is causal (via
    right padding), and one depthwise convolution that is applied only within chunks, that we
    multiply by a scaling factor which depends on the position within the chunk.
    """

    def __init__(
        self, num_channels: int, kernel_size: int, right_context: int, device: torch.device,
    ) -> None:
        """
        ChunkCausalDepthwiseConv1d initialization.

        Parameters
        ----------
        num_channels : int
            The number of input channels.
        kernel_size : int
            The kernel size for chunkwise convolution. The causal convolution kernel size is
            the half of this original value. Should be an odd number.
        right_context : int
            The module look ahead future context, used to update module left cache correctly.
        device : torch.device
            The device used to store the layer weights.
            Either torch.device("cpu") or torch.device("cuda").
        """

        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(
                'Kernel size for ChunkCausalDepthwiseConv1d convolution '
                f'module should be an odd number, but got {kernel_size}.',
            )

        self.kernel_size = kernel_size
        self.right_context = right_context

        self.causal_conv = torch.nn.Conv1d(
            num_channels, num_channels, (kernel_size + 1) // 2, groups=num_channels, device=device,
        )

        self.chunkwise_conv = torch.nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=num_channels,
            device=device,
        )

        # First row is correction factors added to the scale near the left edge of the chunk,
        # second row is correction factors added to the scale near the right edge of the chunk,
        # both of these are added to a default scale of 1.0.
        self.chunkwise_conv_scale = torch.nn.Parameter(
            torch.zeros(2, num_channels, kernel_size, dtype=torch.float32, device=device),
        )

    def forward(
        self, x: torch.Tensor, left_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Does a forward pass of the ChunkCausalDepthwiseConv1d module. Returns processed tensor
        of the same shape as input and updated cached convolution tensor of the left context.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            The input float tensor of shape (1, num_channels, seq_len). The module input.
        left_cache : torch.Tensor[torch.float32]
            A cached convolution tensor of the left context
            of shape (1, num_channels, left_cache_len).

        Returns
        -------
        tuple[torch.Tensor[torch.float32], torch.Tensor[torch.float32]]
            A tuple of two float tensors:
            - module output of shape (1, num_channels, seq_len).
              A tensor with the same shape as input x.
            - updated cached convolution tensor of the left context
              of shape (1, num_channels, left_cache_len).
        """

        seq_len = x.size(2)

        x_chunk = self.chunkwise_conv(x)  # does not change shape

        x = torch.cat((left_cache, x), dim=2)  # Pad with left cache
        left_cache = x[
            :,
            :,
            x.size(2) - self.right_context - left_cache.size(2):
            x.size(2) - self.right_context,
        ]  # Update cache

        x_causal = self.causal_conv(x)

        if seq_len < self.kernel_size:
            left_edge = self.chunkwise_conv_scale[:1, :, :seq_len]
            right_edge = self.chunkwise_conv_scale[1:, :, self.kernel_size - seq_len:]
        else:
            pad = torch.zeros(
                1, self.chunkwise_conv_scale.size(1), seq_len - self.kernel_size,
                dtype=torch.float32,
                device=self.chunkwise_conv_scale.device,
            )
            left_edge = torch.cat((self.chunkwise_conv_scale[:1], pad), dim=2)
            right_edge = torch.cat((pad, self.chunkwise_conv_scale[1:]), dim=2)

        chunk_scale = 1.0 + left_edge + right_edge

        x = x_chunk * chunk_scale + x_causal

        return x, left_cache


class SwooshL(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass and returns Swoosh-L activation.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            A float tensor of an arbitrary shape (*). The module input.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of an arbitrary shape (*). A Swoosh-L activation output tensor
            of the same shape as input x.
        """

        logaddexp = torch.clamp(x - 4.0, min=0.0) + torch.log1p(torch.exp(-torch.abs(x - 4.0)))

        return logaddexp - 0.08 * x - 0.035


class SwooshR(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Does a forward pass and returns Swoosh-R activation.

        Parameters
        ----------
        x : torch.Tensor[torch.float32]
            A float tensor of an arbitrary shape (*). The module input.

        Returns
        -------
        torch.Tensor[torch.float32]
            A float tensor of an arbitrary shape (*). A Swoosh-R activation output tensor
            of the same shape as input x.
        """

        logaddexp = torch.clamp(x - 1.0, min=0.0) + torch.log1p(torch.exp(-torch.abs(x - 1.0)))

        return logaddexp - 0.08 * x - 0.313261687


def _test_whiten():
    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"_test_whiten(): proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        m = Whiten(
            1, 5.0, prob=1.0, grad_scale=0.1  # num_groups  # whitening_limit,
        )  # grad_scale

        for _ in range(4):
            y = m(x)

        y_grad = torch.randn_like(x)
        y.backward(gradient=y_grad)

        if proportion < 0.2:
            assert torch.allclose(x.grad, y_grad)
        elif proportion > 1.0:
            assert not torch.allclose(x.grad, y_grad)


def _test_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * ((2.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))) - 1.0)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        probs.numel(),
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.95,
        min_abs=0.0,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_sign: x = ", x)
    print("_test_balancer_sign: y grad = ", y_grad)
    print("_test_balancer_sign: x grad = ", x.grad)


def _test_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        magnitudes.numel(),
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        min_abs=0.2,
        max_abs=0.7,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_magnitude: x = ", x)
    print("_test_balancer_magnitude: y grad = ", y_grad)
    print("_test_balancer_magnitude: x grad = ", x.grad)


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = DoubleSwish()

    tol = (1.2 - (-0.043637)) / 255.0
    torch.autograd.gradcheck(m, x, atol=tol)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swooshl_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshL()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swooshr_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshR()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_softmax():
    a = torch.randn(2, 10, dtype=torch.float64)
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True
    a.softmax(dim=1)[:, 0].sum().backward()
    print("a grad = ", a.grad)
    softmax(b, dim=1)[:, 0].sum().backward()
    print("b grad = ", b.grad)
    assert torch.allclose(a.grad, b.grad)


def _test_piecewise_linear():
    p = PiecewiseLinear((0, 10.0))
    for x in [-100, 0, 100]:
        assert p(x) == 10.0
    p = PiecewiseLinear((0, 10.0), (1, 0.0))
    for x, y in [(-100, 10.0), (0, 10.0), (0.5, 5.0), (1, 0.0), (2, 0.0)]:
        print("x, y = ", x, y)
        assert p(x) == y, (x, p(x), y)

    q = PiecewiseLinear((0.5, 15.0), (0.6, 1.0))
    x_vals = [-1.0, 0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 1.0, 2.0]
    pq = p.max(q)
    for x in x_vals:
        y1 = max(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p.min(q)
    for x in x_vals:
        y1 = min(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p + q
    for x in x_vals:
        y1 = p(x) + q(x)
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001


def _test_activation_dropout_and_linear():
    in_channels = 20
    out_channels = 30

    for bias in [True, False]:
        # actually we don't test for dropout_p != 0.0 because forward functions will give
        # different answers.  This is because we are using the k2 implementation of
        # swoosh_l an swoosh_r inside SwooshL() and SwooshR(), and they call randn()
        # internally, messing up the random state.
        for dropout_p in [0.0]:
            for activation in ["SwooshL", "SwooshR"]:
                m1 = nn.Sequential(
                    SwooshL() if activation == "SwooshL" else SwooshR(),
                    Dropout3(p=dropout_p, shared_dim=-1),
                    ScaledLinear(
                        in_channels, out_channels, bias=bias, initial_scale=0.5
                    ),
                )
                m2 = ActivationDropoutAndLinear(
                    in_channels,
                    out_channels,
                    bias=bias,
                    initial_scale=0.5,
                    activation=activation,
                    dropout_p=dropout_p,
                )
                with torch.no_grad():
                    m2.weight[:] = m1[2].weight
                    if bias:
                        m2.bias[:] = m1[2].bias
                # make sure forward gives same result.
                x1 = torch.randn(10, in_channels)
                x1.requires_grad = True

                # TEMP.
                assert torch.allclose(
                    SwooshRFunction.apply(x1), SwooshRForward(x1), atol=1.0e-03
                )

                x2 = x1.clone().detach()
                x2.requires_grad = True
                seed = 10
                torch.manual_seed(seed)
                y1 = m1(x1)
                y_grad = torch.randn_like(y1)
                y1.backward(gradient=y_grad)
                torch.manual_seed(seed)
                y2 = m2(x2)
                y2.backward(gradient=y_grad)

                print(
                    f"bias = {bias}, dropout_p = {dropout_p}, activation = {activation}"
                )
                print("y1 = ", y1)
                print("y2 = ", y2)
                assert torch.allclose(y1, y2, atol=0.02)
                assert torch.allclose(m1[2].weight.grad, m2.weight.grad, atol=1.0e-05)
                if bias:
                    assert torch.allclose(m1[2].bias.grad, m2.bias.grad, atol=1.0e-05)
                print("x1.grad = ", x1.grad)
                print("x2.grad = ", x2.grad)

                def isclose(a, b):
                    # return true if cosine similarity is > 0.9.
                    return (a * b).sum() > 0.9 * (
                        (a**2).sum() * (b**2).sum()
                    ).sqrt()

                # the SwooshL() implementation has a noisy gradient due to 1-byte
                # storage of it.
                assert isclose(x1.grad, x2.grad)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_piecewise_linear()
    _test_softmax()
    _test_whiten()
    _test_balancer_sign()
    _test_balancer_magnitude()
    _test_double_swish_deriv()
    _test_swooshr_deriv()
    _test_swooshl_deriv()
    _test_activation_dropout_and_linear()
