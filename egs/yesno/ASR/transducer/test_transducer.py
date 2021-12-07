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
    python ./transducer/test_transducer.py
"""


import k2
import torch
from transducer.decoder import Decoder
from transducer.encoder import Tdnn
from transducer.joiner import Joiner
from transducer.model import Transducer


def test_transducer():
    # encoder params
    input_dim = 10
    output_dim = 20

    # decoder params
    vocab_size = 3
    blank_id = 0
    embedding_dim = 128
    num_layers = 2

    encoder = Tdnn(input_dim, output_dim)

    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        blank_id=blank_id,
        num_layers=num_layers,
        hidden_dim=output_dim,
        embedding_dropout=0.0,
        rnn_dropout=0.0,
    )

    joiner = Joiner(output_dim, vocab_size)
    transducer = Transducer(encoder=encoder, decoder=decoder, joiner=joiner)

    y = k2.RaggedTensor([[1, 2, 1], [1, 1, 1, 2, 1]])
    N = y.dim0
    T = 50

    x = torch.rand(N, T, input_dim)
    x_lens = torch.randint(low=30, high=T, size=(N,), dtype=torch.int32)
    x_lens[0] = T

    loss = transducer(x, x_lens, y)
    print(loss)


def main():
    test_transducer()


if __name__ == "__main__":
    main()
