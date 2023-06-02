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
from typing import List, Tuple

import torch
import torch.nn as nn
from scaling import ActivationBalancer, BasicNorm, Whiten
from zipformer import PoolingModule


class PoolingModuleNoProj(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        cached_len: torch.Tensor,
        cached_avg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (T, N, C)
          cached_len:
            A tensor of shape (N,)
          cached_avg:
            A tensor of shape (N, C)
        Returns:
          Return a tuple containing:
            - new_x
            - new_cached_len
            - new_cached_avg
        """
        x = x.cumsum(dim=0)  # (T, N, C)
        x = x + (cached_avg * cached_len.unsqueeze(1)).unsqueeze(0)
        # Cumulated numbers of frames from start
        cum_mask = torch.arange(1, x.size(0) + 1, device=x.device)
        cum_mask = cum_mask.unsqueeze(1) + cached_len.unsqueeze(0)  # (T, N)
        pooling_mask = (1.0 / cum_mask).unsqueeze(2)
        # now pooling_mask: (T, N, 1)
        x = x * pooling_mask  # (T, N, C)

        cached_len = cached_len + x.size(0)
        cached_avg = x[-1]

        return x, cached_len, cached_avg


class PoolingModuleWithProj(nn.Module):
    def __init__(self, proj: torch.nn.Module):
        super().__init__()
        self.proj = proj
        self.pooling = PoolingModuleNoProj()

    def forward(
        self,
        x: torch.Tensor,
        cached_len: torch.Tensor,
        cached_avg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (T, N, C)
          cached_len:
            A tensor of shape (N,)
          cached_avg:
            A tensor of shape (N, C)
        Returns:
          Return a tuple containing:
            - new_x
            - new_cached_len
            - new_cached_avg
        """
        x, cached_len, cached_avg = self.pooling(x, cached_len, cached_avg)
        return self.proj(x), cached_len, cached_avg

    def streaming_forward(
        self,
        x: torch.Tensor,
        cached_len: torch.Tensor,
        cached_avg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (T, N, C)
          cached_len:
            A tensor of shape (N,)
          cached_avg:
            A tensor of shape (N, C)
        Returns:
          Return a tuple containing:
            - new_x
            - new_cached_len
            - new_cached_avg
        """
        x, cached_len, cached_avg = self.pooling(x, cached_len, cached_avg)
        return self.proj(x), cached_len, cached_avg


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
    assert isinstance(basic_norm, BasicNorm), type(basic_norm)
    norm = NonScaledNorm(
        num_channels=basic_norm.num_channels,
        eps_exp=basic_norm.eps.data.exp().item(),
        channel_dim=basic_norm.channel_dim,
    )
    return norm


def convert_pooling_module(pooling: PoolingModule) -> PoolingModuleWithProj:
    assert isinstance(pooling, PoolingModule), type(pooling)
    return PoolingModuleWithProj(proj=pooling.proj)


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
        elif isinstance(m, PoolingModule) and is_pnnx:
            d[name] = convert_pooling_module(m)

    for k, v in d.items():
        if "." in k:
            parent, child = k.rsplit(".", maxsplit=1)
            setattr(get_submodule(model, parent), child, v)
        else:
            setattr(model, k, v)

    return model
