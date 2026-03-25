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

import logging
import warnings
from typing import Tuple, Optional

import torch
from zapformer_modules import (
    ScaledLinear,
    SwashL,
    SwashR,
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
        causal: bool = False,
    ):
        super().__init__()
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        self.causal = causal
        hidden_channels = channels * hidden_ratio

        if not causal:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        else:
            padding = (0, kernel_size[1] // 2)
            self.left_pad = kernel_size[0] - 1

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=padding,
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

        if self.causal:
            x = nn.functional.pad(x, (0, 0, self.left_pad, 0))
        x = self.depthwise_conv(x)
        assert x.shape == bypass.shape, (x.shape, bypass.shape)

        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)
            cache: (batch_size, num_channels, left_pad, num_freqs)

        Returns:
            - The returned value has the same shape as x.
            - Updated cache.
        """
        bypass = x

        # Pad left side with cache, and update cache
        assert cache.size(2) == self.left_pad
        x = torch.cat([cache, x], dim=2)
        cache = x[:, :, -self.left_pad :, :]

        x = self.depthwise_conv(x)
        assert x.shape == bypass.shape, (x.shape, bypass.shape)

        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x

        return x, cache


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
        causal: bool = False,
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
        # The AddNoise module is there to prevent the gradients
        # w.r.t. the weight or bias of the first Conv2d module in self.conv from
        # getting too large.   The justification in my mind for why this might work
        # is that the first Conv2d module increases the dimension of the input quite a bit,
        # so its output lives in a linear subspace; and there may  in principle be quite large gradients
        # in directions not in this subspace, without affecting the model quality.
        # so by adding a little noise we force the model to "ignore" directions not in this subspace,
        # as much as possible, which will tend to avoid very large gradients.  The reason the
        # large gradients are a problem is because of float16 training with GradScaler, the infinities will
        # be detected and will make it scale the grads by a smaller amount..
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=(0, 1),  # (time, freq)
            ),
            AddNoise(rel_noise_scale=5.0e-03),  # this AddNoise
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
        self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7), causal=causal)

        # (in_channels-3)//4
        self.out_width = (((in_channels - 1) // 2) - 1) // 2
        self.layer3_channels = layer3_channels

        # scale it up a bit, else the output is quite small.
        self.out = ScaledLinear(self.out_width * layer3_channels, out_channels)

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

        x = self.out(x)
        # Now x is of shape (N, (T-7)//2, odim)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            x_lens = (x_lens - 7) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_lens = (x_lens - 7) // 2

        key_padding_mask = torch.arange(0, x.shape[1], device=x.device) >= x_lens.unsqueeze(-1)

        assert x.size(1) == x_lens.max().item(), (x.size(1), x_lens.max())

        return 0.15 * x, x_lens

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        cache: Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
          cache:
            The cached left padding for ConvNeXt module, of shape (batch_size, num_channels, left_pad, num_freqs)

        Returns:
          - a tensor of shape (N, (T-7)//2, odim)
          - output lengths, of shape (batch_size,)
          - updated cache
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)

        # T' = (T-7)//2
        x = self.conv(x)

        x, cache = self.convnext.streaming_forward(x, cache=cache)

        # Now x is of shape (N, odim, T', ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()

        x = x.transpose(1, 2).reshape(b, t, c * f)
        # now x: (N, T', out_width * layer3_channels))

        x = self.out(x)
        # Now x is of shape (N, T', odim)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            x_lens = (x_lens - 7) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_lens = (x_lens - 7) // 2

        assert x.size(1) == x_lens.max().item(), (x.shape, x_lens.max())

        return 0.15 * x, x_lens, cache

    @torch.jit.export
    def get_init_cache(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        """Get initial states for Conv2dSubsampling module.
        It is the cached left padding for ConvNeXt module,
        of shape (batch_size, num_channels, left_pad, num_freqs)
        """
        left_pad = self.convnext.left_pad
        freq = self.out_width
        channels = self.layer3_channels
        cache = torch.zeros(batch_size, channels, left_pad, freq, device=device)

        return cache


def _test_conv2d_subsampling_streaming():
    logging.info("Testing Conv2dSubsampling streaming equivalence...")

    batch_size = 2
    idim = 80
    odim = 256

    model = Conv2dSubsampling(
        in_channels=idim,
        out_channels=odim,
        causal=True
    )

    model.eval()

    out_chunk_size = 32
    in_chunk_size = out_chunk_size * 2 + 7
    in_shift = out_chunk_size * 2

    num_chunks = 10

    seq_len = num_chunks * in_shift + 7

    x_full = torch.randn(batch_size, seq_len, idim)
    x_lens_full = torch.full((batch_size,), seq_len, dtype=torch.int64)

    with torch.no_grad():
        out_full, out_lens_full = model(x_full, x_lens_full)

        cache = model.get_init_cache(batch_size=batch_size)

        out_chunks = []
        out_offset = 0

        for i in range(num_chunks):
            start = i * in_shift
            end = start + in_chunk_size
            x_chunk = x_full[:, start:end, :]
            x_lens_chunk = torch.full((batch_size,), in_chunk_size, dtype=torch.int64)

            out_chunk, out_lens_chunk, cache = model.streaming_forward(
                x_chunk, x_lens_chunk, cache
            )
            out_chunks.append(out_chunk)

            out_chunk_len = out_chunk.shape[1]
            expected_out = out_full[:, out_offset : out_offset + out_chunk_len, :]

            diff_chunk = torch.max(torch.abs(expected_out - out_chunk))
            logging.info(f"Chunk {i+1} | Input: {x_chunk.shape} -> Output: {out_chunk.shape} | Max diff: {diff_chunk}")

            assert torch.allclose(expected_out, out_chunk, atol=1e-4), f"Chunk {i+1} mismatch! max diff: {diff_chunk}"
            out_offset += out_chunk_len

        out_stream_cat = torch.cat(out_chunks, dim=1)
        diff_total = torch.max(torch.abs(out_full - out_stream_cat))
        logging.info(f"Total Max Diff between full forward and streaming: {diff_total}")
        assert torch.allclose(out_full, out_stream_cat, atol=1e-4), "Total outputs do not match!"

    logging.info("Passed")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_conv2d_subsampling_streaming()
