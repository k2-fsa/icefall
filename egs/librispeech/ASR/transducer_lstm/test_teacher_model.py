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
    python ./transducer_lstm/test_teacher_model.py
"""

import warnings

import torch
from teacher_model import get_teacher_model


def test_teacher_model():
    model = get_teacher_model()
    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of encoder model parameters: {num_param}")

    N = 3
    T = 500
    C = 80

    x = torch.rand(N, T, C)
    x_lens = torch.tensor([100, 500, 300])

    y, y_lens = model.encoder(x, x_lens)
    print(y.shape)
    expected_y_lens = (((x_lens - 1) >> 1) - 1) >> 1

    assert torch.all(torch.eq(y_lens, expected_y_lens)), (
        y_lens,
        expected_y_lens,
    )


def main():
    test_teacher_model()


if __name__ == "__main__":
    main()
