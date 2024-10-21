# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file at https://github.com/facebookresearch/encodec/blob/main/LICENSE
"""Normalization modules."""

from typing import List, Union

import einops
import torch
from torch import nn


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return
