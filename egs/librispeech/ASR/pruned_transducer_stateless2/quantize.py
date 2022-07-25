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

import copy
import re

import torch
from scaling import ScaledLinear


def _get_weight(self: torch.nn.Linear):
    return self.weight


def _get_bias(self: torch.nn.Linear):
    return self.bias


def scaled_linear_to_linear(scaled_linear: ScaledLinear) -> torch.nn.Linear:
    """Convert a ScaledLinear layer to a Linear layer.

    ScaledLinear layer is used for training. However, during inference
    we only need a Linear layer.

    You will need this function when you want to do quantization since
    ScaledLinear cannot be quantized by PyTorch.

    Args:
      scaled_linear:
        An instance of ScaledLinear.
    Returns:
      Return an instance of torch.nn.Linear. It satisfies

        scaled_linear(x) == linear(x)

      for any given input tensor x.
    """
    if not hasattr(torch.nn.Linear, "get_weight"):
        torch.nn.Linear.get_weight = _get_weight
        torch.nn.Linear.get_bias = _get_bias

    assert isinstance(scaled_linear, ScaledLinear), type(scaled_linear)
    weight = scaled_linear.get_weight()
    bias = scaled_linear.get_bias()
    has_bias = bias is not None

    linear = torch.nn.Linear(
        in_features=scaled_linear.in_features,
        out_features=scaled_linear.out_features,
        bias=has_bias,
        device=weight.device,
    )
    linear.weight.data.copy_(weight)

    if has_bias:
        linear.bias.data.copy_(bias)

    return linear


def convert_scaled_linear(model: torch.nn.Module, inplace: bool = False):
    """Convert **all** ScaledLinear layers in a model to Linear layers.

    Args:
      model:
        The input model to be converted.
      inplace:
        If True, the input model is modified **inplace**.
        If False, the input model is copied and we modify the copy.
    Returns:
      Return the converted model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    d = {}
    excluded_patterns = r"self_attn\.(in|out)_proj"
    p = re.compile(excluded_patterns)
    for name, m in model.named_modules():
        if isinstance(m, ScaledLinear):
            if p.search(name) is not None:
                continue
            d[name] = scaled_linear_to_linear(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(model.get_submodule(parent), child, v)
        else:
            setattr(model, k, v)

    return model


def dynamic_quantize(
    model: torch.nn.Module,
    inplace: bool = False,
) -> torch.nn.Module:
    """Apply post-training dynamic quantization to a given model.

    It is also known as post-training weight-only quantization.
    Weights are quantized to tensors of dtype torch.qint8.

    Only nn.Linear layers are quantized at present.

    Args:
      model:
        The model to be quantized.
      inplace:
        If True, the passed model is modified inplace.
        If False, the passed model is copied and we modify the copied model.
    """
    converted_model = convert_scaled_linear(model)
    q_model = torch.quantization.quantize_dynamic(
        model=converted_model,
        qconfig_spec={torch.nn.Linear},
        dtype=torch.qint8,
        inplace=inplace,
    )
    return q_model
