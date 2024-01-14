#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./transducer/test_conformer.py
"""

import torch
from conformer import Conformer


def test_conformer():
    output_dim = 1024
    conformer = Conformer(
        num_features=80,
        output_dim=output_dim,
        subsampling_factor=4,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=12,
    )
    N = 3
    T = 100
    C = 80
    x = torch.randn(N, T, C)
    x_lens = torch.tensor([50, 100, 80])
    logits, logit_lens = conformer(x, x_lens)

    expected_T = ((T - 1) // 2 - 1) // 2
    assert logits.shape == (N, expected_T, output_dim)
    assert logit_lens.max().item() == expected_T
    print(logits.shape)
    print(logit_lens)


def main():
    test_conformer()


if __name__ == "__main__":
    main()
