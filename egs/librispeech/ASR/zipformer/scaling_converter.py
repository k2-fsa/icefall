# Copyright    2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
from typing import List, Tuple

import torch
import torch.nn as nn
from scaling import (
    Balancer,
    Dropout3,
    ScaleGrad,
    SwooshL,
    SwooshLOnnx,
    SwooshR,
    SwooshROnnx,
    Whiten,
)
from zipformer import CompactRelPositionalEncoding


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
        elif is_onnx and isinstance(m, CompactRelPositionalEncoding):
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
