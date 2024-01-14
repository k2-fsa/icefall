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
    python ./transducer_stateless_multi_datasets/test_decoder.py
"""

import torch
from decoder import Decoder


def test_decoder():
    vocab_size = 3
    blank_id = 0
    embedding_dim = 128
    context_size = 4

    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        blank_id=blank_id,
        context_size=context_size,
    )
    N = 100
    U = 20
    x = torch.randint(low=0, high=vocab_size, size=(N, U))
    y = decoder(x)
    assert y.shape == (N, U, embedding_dim)

    # for inference
    x = torch.randint(low=0, high=vocab_size, size=(N, context_size))
    y = decoder(x, need_pad=False)
    assert y.shape == (N, 1, embedding_dim)


def main():
    test_decoder()


if __name__ == "__main__":
    main()
