# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file provides functions to convert `ScaledLinear`, `ScaledConv1d`,
and `ScaledConv2d` to their non-scaled counterparts: `nn.Linear`, `nn.Conv1d`,
and `nn.Conv2d`.

The scaled version are required only in the training time. It simplifies our
life by converting them their non-scaled version during inference time.
"""

import copy
import re

import torch
import torch.nn as nn
from scaling import ScaledConv1d, ScaledConv2d, ScaledLinear


def _get_weight(self: torch.nn.Linear):
    return self.weight


def _get_bias(self: torch.nn.Linear):
    return self.bias


def scaled_linear_to_linear(scaled_linear: ScaledLinear) -> nn.Linear:
    """Convert an instance of ScaledLinear to nn.Linear.

    Args:
      scaled_linear:
        The layer to be converted.
    Returns:
      Return a linear layer. It satisfies:

        scaled_linear(x) == linear(x)

      for any given input tensor `x`.
    """
    assert isinstance(scaled_linear, ScaledLinear), type(scaled_linear)

    #  if not hasattr(torch.nn.Linear, "get_weight"):
    #      torch.nn.Linear.get_weight = _get_weight
    #      torch.nn.Linear.get_bias = _get_bias

    weight = scaled_linear.get_weight()
    bias = scaled_linear.get_bias()
    has_bias = bias is not None

    linear = torch.nn.Linear(
        in_features=scaled_linear.in_features,
        out_features=scaled_linear.out_features,
        bias=True,  # otherwise, it throws errors when converting to PNNX format.
        device=weight.device,
    )
    linear.weight.data.copy_(weight)

    if has_bias:
        linear.bias.data.copy_(bias)
    else:
        linear.bias.data.zero_()

    return linear


def scaled_conv1d_to_conv1d(scaled_conv1d: ScaledConv1d) -> nn.Conv1d:
    """Convert an instance of ScaledConv1d to nn.Conv1d.

    Args:
      scaled_conv1d:
        The layer to be converted.
    Returns:
      Return an instance of nn.Conv1d that has the same `forward()` behavior
      of the given `scaled_conv1d`.
    """
    assert isinstance(scaled_conv1d, ScaledConv1d), type(scaled_conv1d)

    weight = scaled_conv1d.get_weight()
    bias = scaled_conv1d.get_bias()
    has_bias = bias is not None

    conv1d = nn.Conv1d(
        in_channels=scaled_conv1d.in_channels,
        out_channels=scaled_conv1d.out_channels,
        kernel_size=scaled_conv1d.kernel_size,
        stride=scaled_conv1d.stride,
        padding=scaled_conv1d.padding,
        dilation=scaled_conv1d.dilation,
        groups=scaled_conv1d.groups,
        bias=scaled_conv1d.bias is not None,
        padding_mode=scaled_conv1d.padding_mode,
    )

    conv1d.weight.data.copy_(weight)
    if has_bias:
        conv1d.bias.data.copy_(bias)

    return conv1d


def scaled_conv2d_to_conv2d(scaled_conv2d: ScaledConv2d) -> nn.Conv2d:
    """Convert an instance of ScaledConv2d to nn.Conv2d.

    Args:
      scaled_conv2d:
        The layer to be converted.
    Returns:
      Return an instance of nn.Conv2d that has the same `forward()` behavior
      of the given `scaled_conv2d`.
    """
    assert isinstance(scaled_conv2d, ScaledConv2d), type(scaled_conv2d)

    weight = scaled_conv2d.get_weight()
    bias = scaled_conv2d.get_bias()
    has_bias = bias is not None

    conv2d = nn.Conv2d(
        in_channels=scaled_conv2d.in_channels,
        out_channels=scaled_conv2d.out_channels,
        kernel_size=scaled_conv2d.kernel_size,
        stride=scaled_conv2d.stride,
        padding=scaled_conv2d.padding,
        dilation=scaled_conv2d.dilation,
        groups=scaled_conv2d.groups,
        bias=scaled_conv2d.bias is not None,
        padding_mode=scaled_conv2d.padding_mode,
    )

    conv2d.weight.data.copy_(weight)
    if has_bias:
        conv2d.bias.data.copy_(bias)

    return conv2d


def convert_scaled_to_non_scaled(model: nn.Module, inplace: bool = False):
    """Convert `ScaledLinear`, `ScaledConv1d`, and `ScaledConv2d`
    in the given modle to their unscaled version `nn.Linear`, `nn.Conv1d`,
    and `nn.Conv2d`.

    Args:
      model:
        The model to be converted.
      inplace:
        If True, the input model is modified inplace.
        If False, the input model is copied and we modify the copied version.
    Return:
      Return a model without scaled layers.
    """
    if not inplace:
        model = copy.deepcopy(model)

    excluded_patterns = r"self_attn\.(in|out)_proj"
    p = re.compile(excluded_patterns)

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, ScaledLinear):
            if p.search(name) is not None:
                continue
            d[name] = scaled_linear_to_linear(m)
        elif isinstance(m, ScaledConv1d):
            d[name] = scaled_conv1d_to_conv1d(m)
        elif isinstance(m, ScaledConv2d):
            d[name] = scaled_conv2d_to_conv2d(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(model.get_submodule(parent), child, v)
        else:
            setattr(model, k, v)

    return model
