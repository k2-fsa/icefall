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
    python ./pruned_transducer_stateless_multi_datasets/test_model.py
"""


import k2
import torch
from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from model import Transducer


def test_model():
    # encoder params
    input_dim = 10
    attention_dim = 512

    # decoder params
    vocab_size = 3
    embedding_dim = 512
    blank_id = 0
    context_size = 2

    joiner_dim = 1024

    encoder = Conformer(
        num_features=input_dim,
        subsampling_factor=4,
        d_model=attention_dim,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=12,
    )

    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        blank_id=blank_id,
        context_size=context_size,
    )

    joiner = Joiner(joiner_dim, vocab_size)
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
    test_model()


if __name__ == "__main__":
    main()
