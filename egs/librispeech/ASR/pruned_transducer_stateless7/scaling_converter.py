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
This file replaces various modules in a model.
Specifically, ActivationBalancer is replaced with an identity operator;
Whiten is also replaced with an identity operator;
BasicNorm is replaced by a module with `exp` removed.
"""

import copy
from typing import List

import torch
import torch.nn as nn
from scaling import ActivationBalancer, BasicNorm, Whiten


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


def convert_basic_norm(basic_norm: BasicNorm) -> NonScaledNorm:
    assert isinstance(basic_norm, BasicNorm), type(BasicNorm)
    norm = NonScaledNorm(
        num_channels=basic_norm.num_channels,
        eps_exp=basic_norm.eps.data.exp().item(),
        channel_dim=basic_norm.channel_dim,
    )
    return norm


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
):
    """
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

    d = {}
    for name, m in model.named_modules():
        if isinstance(m, BasicNorm):
            d[name] = convert_basic_norm(m)
        elif isinstance(m, (ActivationBalancer, Whiten)):
            d[name] = nn.Identity()

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    return model
