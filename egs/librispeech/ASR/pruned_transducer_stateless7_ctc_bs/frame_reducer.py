#!/usr/bin/env python3
#
# Copyright      2022  Xiaomi Corp.        (authors: Yifan Yang,
#                                                    Zengwei Yao)
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

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from icefall.utils import make_pad_mask


class FrameReducer(nn.Module):
    """The encoder output is first used to calculate
    the CTC posterior probability; then for each output frame,
    if its blank posterior is bigger than some thresholds,
    it will be simply discarded from the encoder output.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        ctc_output: torch.Tensor,
        blank_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:
              The shared encoder output with shape [N, T, C].
            x_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `x` before padding.
            ctc_output:
              The CTC output with shape [N, T, vocab_size].
            blank_id:
              The ID of the blank symbol.
        Returns:
            x_fr:
              The frame reduced encoder output with shape [N, T', C].
            x_lens_fr:
              A tensor of shape (batch_size,) containing the number of frames in
              `x_fr` before padding.
        """

        padding_mask = make_pad_mask(x_lens)
        non_blank_mask = (ctc_output[:, :, blank_id] < math.log(0.9)) * (~padding_mask)
        T_range = torch.arange(x.shape[1], device=x.device)

        frames_list: List[torch.Tensor] = []
        lens_list: List[int] = []
        for i in range(x.shape[0]):
            indexes = torch.masked_select(
                T_range,
                non_blank_mask[i],
            )
            frames = x[i][indexes]
            frames_list.append(frames)
            lens_list.append(frames.shape[0])
        x_fr = pad_sequence(frames_list).transpose(0, 1)
        x_lens_fr = torch.tensor(lens_list).to(device=x.device)

        return x_fr, x_lens_fr
