#!/usr/bin/env python3
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
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./transducer_emformer/test_emformer.py
"""

import warnings

import torch
from emformer import Emformer


def test_emformer():
    N = 3
    T = 300
    C = 80

    output_dim = 500

    encoder = Emformer(
        num_features=C,
        output_dim=output_dim,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=12,
        segment_length=16,
        left_context_length=120,
        right_context_length=4,
        vgg_frontend=True,
    )

    x = torch.rand(N, T, C)
    x_lens = torch.randint(100, T, (N,))
    x_lens[0] = T

    y, y_lens = encoder(x, x_lens)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (y_lens == ((x_lens - 1) // 2 - 1) // 2).all()
    assert x.size(0) == x.size(0)
    assert y.size(1) == max(y_lens)
    assert y.size(2) == output_dim

    num_param = sum([p.numel() for p in encoder.parameters()])
    print(f"Number of encoder parameters: {num_param}")


def main():
    test_emformer()


if __name__ == "__main__":
    torch.manual_seed(20220329)
    main()
