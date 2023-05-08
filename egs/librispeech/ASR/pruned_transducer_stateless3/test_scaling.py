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
    python ./pruned_transducer_stateless3/test_scaling.py
"""

import torch
from scaling import ActivationBalancer, ScaledConv1d, ScaledConv2d


def test_scaled_conv1d():
    for bias in [True, False]:
        conv1d = ScaledConv1d(
            3,
            6,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        torch.jit.script(conv1d)


def test_scaled_conv2d():
    for bias in [True, False]:
        conv2d = ScaledConv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        torch.jit.script(conv2d)


def main():
    test_scaled_conv1d()
    test_scaled_conv2d()


if __name__ == "__main__":
    main()
