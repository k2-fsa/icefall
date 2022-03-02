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
    python ./transducer_stateless/test_joiner.py
"""

import torch
from joiner import Joiner


def test_joiner():
    device = torch.device("cpu")
    input_dim = 3
    output_dim = 5
    joiner = Joiner(input_dim, output_dim)
    joiner.to(device)

    encoder_out = torch.rand(3, 10, input_dim, device=device)
    decoder_out = torch.rand(3, 8, input_dim, device=device)

    encoder_out_len = torch.tensor([5, 10, 3], device=device)
    decoder_out_len = torch.tensor([6, 8, 7], device=device)

    out = joiner(
        encoder_out=encoder_out,
        decoder_out=decoder_out,
        encoder_out_len=encoder_out_len,
        decoder_out_len=decoder_out_len,
    )
    assert out.size(0) == (encoder_out_len * decoder_out_len).sum()
    assert out.size(1) == output_dim


def main():
    test_joiner()


if __name__ == "__main__":
    main()
