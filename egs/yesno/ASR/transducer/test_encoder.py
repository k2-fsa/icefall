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

    cd icefall/egs/yesno/ASR
    python ./transducer/test_encoder.py
"""

import torch
from transducer.encoder import Tdnn


def test_encoder():
    input_dim = 10
    output_dim = 20
    encoder = Tdnn(input_dim, output_dim)
    N = 10
    T = 85
    x = torch.rand(N, T, input_dim)
    x_lens = torch.randint(low=30, high=T, size=(N,), dtype=torch.int32)
    logits, logit_lens = encoder(x, x_lens)
    assert logits.shape == (N, T - 26, output_dim)
    assert torch.all(torch.eq(x_lens - 26, logit_lens))


def main():
    test_encoder()


if __name__ == "__main__":
    main()
