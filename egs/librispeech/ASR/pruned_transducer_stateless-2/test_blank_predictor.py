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
    python ./pruned_transducer_stateless_2/test_blank_predictor.py
"""
import torch
from blank_predictor import BlankPredictor


def test_blank_predictor():
    dim = 10
    predictor = BlankPredictor(encoder_out_dim=dim)
    x = torch.rand(4, 3, dim)
    x_lens = torch.tensor([1, 3, 2, 3], dtype=torch.int32)
    y = torch.rand(4, 3)
    loss = predictor(x, x_lens, y)
    print(loss)


def main():
    test_blank_predictor()


if __name__ == "__main__":
    main()
