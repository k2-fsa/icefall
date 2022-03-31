# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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

import torch
import torch.nn as nn

from icefall.utils import make_pad_mask


class BlankPredictor(nn.Module):
    def __init__(self, encoder_out_dim: int):
        """
        Args:
          Output dimension of the encoder network.
        """
        super().__init__()
        self.linear = nn.Linear(in_features=encoder_out_dim, out_features=1)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        soft_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, encoder_out_dim) from the output of
            the encoder network.
          x_lens:
            A 1-D tensor of shape (N,) containing the number of valid frames
            for each element in `x`.
          soft_target:
            A 2-D tensor of shape (N, T) containing the soft label of each frame
            in `x`.
        """
        assert x.ndim == 3, x.shape
        assert soft_target.ndim == 2, soft_target.shape

        assert x.shape[:2] == soft_target.shape[:2], (
            x.shape,
            soft_target.shape,
        )
        logits = self.linear(x).squeeze(-1)
        mask = make_pad_mask(x_lens)

        loss = self.loss_func(logits, soft_target)
        loss.masked_fill_(mask, 0)

        return loss.sum()
