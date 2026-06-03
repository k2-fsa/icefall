# Copyright    2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao)
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

"""
This file replaces various modules in a model.
Specifically, ActivationBalancer is replaced with an identity operator;
Whiten is also replaced with an identity operator;
BasicNorm is replaced by a module with `exp` removed.
"""

import copy
from typing import List

import torch
import torch.nn as nn
from scaling import (
    Balancer,
    ChunkCausalDepthwiseConv1d,
    Dropout3,
    ScaleGrad,
    SwooshL,
    SwooshLOnnx,
    SwooshR,
    SwooshROnnx,
    Whiten,
)
from zipformer import (
    CompactRelPositionalEncoding,
    SimpleDownsample,
)


class NonStreamingChunkCausalDepthwiseConv1d(torch.nn.Module):
    """A non-streaming replacement for ChunkCausalDepthwiseConv1d that avoids
    dynamic-shape torch.zeros and conditionals, making it ONNX-export friendly.

    In non-streaming mode (chunk_size=-1), the entire sequence is one chunk,
    so we simplify the forward pass accordingly.
    """

    def __init__(self, original: ChunkCausalDepthwiseConv1d):
        super().__init__()
        self.causal_conv = original.causal_conv
        self.chunkwise_conv = original.chunkwise_conv
        self.chunkwise_conv_scale = original.chunkwise_conv_scale
        self.kernel_size = original.kernel_size

    def forward(self, x: torch.Tensor, chunk_size: int = -1) -> torch.Tensor:
        (batch_size, num_channels, seq_len) = x.shape
        left_pad = self.kernel_size // 2

        x = torch.nn.functional.pad(x, (left_pad, 0))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)

        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        # seq_len >= kernel_size in non-streaming mode, so we pad with zeros
        t = seq_len - self.kernel_size
        channels = left_edge.shape[0]
        pad = torch.zeros(
            channels, t, device=left_edge.device, dtype=left_edge.dtype
        )
        left_edge = torch.cat((left_edge, pad), dim=-1)
        right_edge = torch.cat((pad, right_edge), dim=-1)
        chunk_scale = 1.0 + (left_edge + right_edge)

        x_chunk = x_chunk * chunk_scale

        return x_chunk + x_causal


# Copied from https://pytorch.org/docs/1.9.0/_modules/torch/nn/modules/module.html#Module.get_submodule  # noqa
# get_submodule was added to nn.Module at v1.9.0
def get_submodule(model, target):
    if target == "":
        return model
    atoms: List[str] = target.split(".")
    mod: torch.nn.Module = model
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(
                mod._get_name() + " has no " "attribute `" + item + "`"
            )
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")
    return mod


def convert_scaled_to_non_scaled(
    model: nn.Module,
    inplace: bool = False,
    is_pnnx: bool = False,
    is_onnx: bool = False,
):
    """
    Args:
      model:
        The model to be converted.
      inplace:
        If True, the input model is modified inplace.
        If False, the input model is copied and we modify the copied version.
      is_pnnx:
        True if we are going to export the model for PNNX.
      is_onnx:
        True if we are going to export the model for ONNX.
    Return:
      Return a model without scaled layers.
    """
    if not inplace:
        model = copy.deepcopy(model)

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, (Balancer, Dropout3, ScaleGrad, Whiten)):
            d[name] = nn.Identity()
        elif is_onnx and isinstance(m, SwooshR):
            d[name] = SwooshROnnx()
        elif is_onnx and isinstance(m, SwooshL):
            d[name] = SwooshLOnnx()
        elif is_onnx and isinstance(m, ChunkCausalDepthwiseConv1d):
            d[name] = torch.jit.script(NonStreamingChunkCausalDepthwiseConv1d(m))
        elif is_onnx and isinstance(
            m,
            (
                CompactRelPositionalEncoding,
                SimpleDownsample,
            ),
        ):
            # We want to recreate the positional encoding vector when
            # the input changes, so we have to use torch.jit.script()
            # to replace torch.jit.trace()
            d[name] = torch.jit.script(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    return model
