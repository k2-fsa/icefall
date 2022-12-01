#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Daniel Povey
#                                                  Fangjun Kuang)
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
    python ./transducer_stateless/test_conformer.py
"""

import torch
from conformer import Conformer


def test_conformer():
    feature_dim = 50
    c = Conformer(num_features=feature_dim, output_dim=256, d_model=128, nhead=4)
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    logits, lengths = c(
        torch.randn(batch_size, seq_len, feature_dim),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    print(logits.shape)
    print(lengths.shape)


def main():
    test_conformer()


if __name__ == "__main__":
    main()
