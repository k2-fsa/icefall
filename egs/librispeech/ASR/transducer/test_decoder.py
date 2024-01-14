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
    python ./transducer/test_decoder.py
"""

import torch
from decoder import Decoder


def test_decoder():
    vocab_size = 3
    blank_id = 0
    embedding_dim = 128
    num_layers = 2
    hidden_dim = 6
    output_dim = 8
    N = 3
    U = 5

    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        blank_id=blank_id,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dropout=0.0,
        rnn_dropout=0.0,
    )
    x = torch.randint(1, vocab_size, (N, U))
    decoder_out, (h, c) = decoder(x)

    assert decoder_out.shape == (N, U, output_dim)
    assert h.shape == (num_layers, N, hidden_dim)
    assert c.shape == (num_layers, N, hidden_dim)

    decoder_out, (h, c) = decoder(x, (h, c))
    assert decoder_out.shape == (N, U, output_dim)
    assert h.shape == (num_layers, N, hidden_dim)
    assert c.shape == (num_layers, N, hidden_dim)


def main():
    test_decoder()


if __name__ == "__main__":
    main()
