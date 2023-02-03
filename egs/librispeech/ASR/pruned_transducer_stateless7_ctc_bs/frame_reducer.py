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
import torch.nn.functional as F

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
              The blank id of ctc_output.
        Returns:
            out:
              The frame reduced encoder output with shape [N, T', C].
            out_lens:
              A tensor of shape (batch_size,) containing the number of frames in
              `out` before padding.
        """

        N, T, C = x.size()

        padding_mask = make_pad_mask(x_lens)
        non_blank_mask = (ctc_output[:, :, blank_id] < math.log(0.9)) * (~padding_mask)

        out_lens = non_blank_mask.sum(dim=1)
        max_len = out_lens.max()
        pad_lens_list = torch.full_like(out_lens, max_len.item()) - out_lens
        max_pad_len = pad_lens_list.max()

        out = F.pad(x, (0, 0, 0, max_pad_len))

        valid_pad_mask = ~make_pad_mask(pad_lens_list)
        total_valid_mask = torch.concat([non_blank_mask, valid_pad_mask], dim=1)

        out = out[total_valid_mask].reshape(N, -1, C)

        return out.to(device=x.device), out_lens.to(device=x.device)


if __name__ == "__main__":
    import time
    from torch.nn.utils.rnn import pad_sequence

    test_times = 10000
    frame_reducer = FrameReducer()

    # non zero case
    x = torch.ones(15, 498, 384, dtype=torch.float32)
    x_lens = torch.tensor([498] * 15, dtype=torch.int64)
    ctc_output = torch.log(torch.randn(15, 498, 500, dtype=torch.float32))
    x_fr, x_lens_fr = frame_reducer(x, x_lens, ctc_output)

    avg_time = 0
    for i in range(test_times):
        delta_time = time.time()
        x_fr, x_lens_fr = frame_reducer(x, x_lens, ctc_output)
        delta_time = time.time() - delta_time
        avg_time += delta_time
    print(x_fr.shape)
    print(x_lens_fr)
    print(avg_time / test_times)

    # all zero case
    x = torch.zeros(15, 498, 384, dtype=torch.float32)
    x_lens = torch.tensor([498] * 15, dtype=torch.int64)
    ctc_output = torch.zeros(15, 498, 500, dtype=torch.float32)

    avg_time = 0
    for i in range(test_times):
        delta_time = time.time()
        x_fr, x_lens_fr = frame_reducer(x, x_lens, ctc_output)
        delta_time = time.time() - delta_time
        avg_time += delta_time
    print(x_fr.shape)
    print(x_lens_fr)
    print(avg_time / test_times)
