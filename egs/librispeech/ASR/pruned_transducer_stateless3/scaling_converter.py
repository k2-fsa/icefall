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
`ScaledConv2d`, and `ScaledEmbedding` to their non-scaled counterparts:
`nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, and `nn.Embedding`.

The scaled version are required only in the training time. It simplifies our
life by converting them to their non-scaled version during inference.
"""

import copy
import re
from typing import List

import torch
import torch.nn as nn
from lstmp import LSTMP
from scaling import (
    ActivationBalancer,
    BasicNorm,
    ScaledConv1d,
    ScaledConv2d,
    ScaledEmbedding,
    ScaledLinear,
    ScaledLSTM,
)


class NonScaledNorm(nn.Module):
    """See BasicNorm for doc"""

    def __init__(
        self,
        num_channels: int,
        eps_exp: float,
        channel_dim: int = -1,  # CAUTION: see documentation.
    ):
        super().__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.eps_exp = eps_exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            assert x.shape[self.channel_dim] == self.num_channels
        scales = (
            torch.mean(x * x, dim=self.channel_dim, keepdim=True) + self.eps_exp
        ).pow(-0.5)
        return x * scales


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

    weight = scaled_linear.get_weight()
    bias = scaled_linear.get_bias()
    has_bias = bias is not None

    linear = torch.nn.Linear(
        in_features=scaled_linear.in_features,
        out_features=scaled_linear.out_features,
        bias=True,  # otherwise, it throws errors when converting to PNNX format
        # device=weight.device,  # Pytorch version before v1.9.0 does not have
        # this argument. Comment out for now, we will
        # see if it will raise error for versions
        # after v1.9.0
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


def scaled_embedding_to_embedding(
    scaled_embedding: ScaledEmbedding,
) -> nn.Embedding:
    """Convert an instance of ScaledEmbedding to nn.Embedding.

    Args:
      scaled_embedding:
        The layer to be converted.
    Returns:
      Return an instance of nn.Embedding that has the same `forward()` behavior
      of the given `scaled_embedding`.
    """
    assert isinstance(scaled_embedding, ScaledEmbedding), type(scaled_embedding)
    embedding = nn.Embedding(
        num_embeddings=scaled_embedding.num_embeddings,
        embedding_dim=scaled_embedding.embedding_dim,
        padding_idx=scaled_embedding.padding_idx,
        scale_grad_by_freq=scaled_embedding.scale_grad_by_freq,
        sparse=scaled_embedding.sparse,
    )
    weight = scaled_embedding.weight
    scale = scaled_embedding.scale

    embedding.weight.data.copy_(weight * scale.exp())

    return embedding


def convert_basic_norm(basic_norm: BasicNorm) -> NonScaledNorm:
    assert isinstance(basic_norm, BasicNorm), type(BasicNorm)
    norm = NonScaledNorm(
        num_channels=basic_norm.num_channels,
        eps_exp=basic_norm.eps.data.exp().item(),
        channel_dim=basic_norm.channel_dim,
    )
    return norm


def scaled_lstm_to_lstm(scaled_lstm: ScaledLSTM) -> nn.LSTM:
    """Convert an instance of ScaledLSTM to nn.LSTM.

    Args:
      scaled_lstm:
        The layer to be converted.
    Returns:
      Return an instance of nn.LSTM that has the same `forward()` behavior
      of the given `scaled_lstm`.
    """
    assert isinstance(scaled_lstm, ScaledLSTM), type(scaled_lstm)
    lstm = nn.LSTM(
        input_size=scaled_lstm.input_size,
        hidden_size=scaled_lstm.hidden_size,
        num_layers=scaled_lstm.num_layers,
        bias=scaled_lstm.bias,
        batch_first=scaled_lstm.batch_first,
        dropout=scaled_lstm.dropout,
        bidirectional=scaled_lstm.bidirectional,
        proj_size=scaled_lstm.proj_size,
    )

    assert lstm._flat_weights_names == scaled_lstm._flat_weights_names
    for idx in range(len(scaled_lstm._flat_weights_names)):
        scaled_weight = scaled_lstm._flat_weights[idx] * scaled_lstm._scales[idx].exp()
        lstm._flat_weights[idx].data.copy_(scaled_weight)

    return lstm


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
    is_onnx: bool = False,
):
    """Convert `ScaledLinear`, `ScaledConv1d`, and `ScaledConv2d`
    in the given modle to their unscaled version `nn.Linear`, `nn.Conv1d`,
    and `nn.Conv2d`.

    Args:
      model:
        The model to be converted.
      inplace:
        If True, the input model is modified inplace.
        If False, the input model is copied and we modify the copied version.
      is_onnx:
        If True, we are going to export the model to ONNX. In this case,
        we will convert nn.LSTM with proj_size to LSTMP.
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
        elif isinstance(m, ScaledEmbedding):
            d[name] = scaled_embedding_to_embedding(m)
        elif isinstance(m, BasicNorm):
            d[name] = convert_basic_norm(m)
        elif isinstance(m, ScaledLSTM):
            if is_onnx:
                d[name] = LSTMP(scaled_lstm_to_lstm(m))
                # See
                # https://github.com/pytorch/pytorch/issues/47887
                #  d[name] = torch.jit.script(LSTMP(scaled_lstm_to_lstm(m)))
            else:
                d[name] = scaled_lstm_to_lstm(m)
        elif isinstance(m, ActivationBalancer):
            d[name] = nn.Identity()

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    return model
