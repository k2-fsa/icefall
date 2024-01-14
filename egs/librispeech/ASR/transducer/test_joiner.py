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
    python ./transducer/test_joiner.py
"""


import torch
from joiner import Joiner


def test_joiner():
    N = 2
    T = 3
    C = 4
    U = 5

    joiner = Joiner(C, 10)

    encoder_out = torch.rand(N, T, C)
    decoder_out = torch.rand(N, U, C)

    joint = joiner(encoder_out, decoder_out)
    assert joint.shape == (N, T, U, 10)


def main():
    test_joiner()


if __name__ == "__main__":
    main()
