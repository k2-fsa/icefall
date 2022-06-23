#!/usr/bin/env python3
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

import torch
from rnn_lm.model import RnnLmModel


def test_rnn_lm_model():
    vocab_size = 4
    model = RnnLmModel(
        vocab_size=vocab_size, embedding_dim=10, hidden_dim=10, num_layers=2
    )
    x = torch.tensor(
        [
            [1, 3, 2, 2],
            [1, 2, 2, 0],
            [1, 2, 0, 0],
        ]
    )
    y = torch.tensor(
        [
            [3, 2, 2, 1],
            [2, 2, 1, 0],
            [2, 1, 0, 0],
        ]
    )
    lengths = torch.tensor([4, 3, 2])
    nll_loss = model(x, y, lengths)
    print(nll_loss)
    """
    tensor([[1.1180, 1.3059, 1.2426, 1.7773],
            [1.4231, 1.2783, 1.7321, 0.0000],
            [1.4231, 1.6752, 0.0000, 0.0000]], grad_fn=<ViewBackward>)
    """


def test_rnn_lm_model_tie_weights():
    model = RnnLmModel(
        vocab_size=10,
        embedding_dim=10,
        hidden_dim=10,
        num_layers=2,
        tie_weights=True,
    )
    assert model.input_embedding.weight is model.output_linear.weight


def main():
    test_rnn_lm_model()
    test_rnn_lm_model_tie_weights()


if __name__ == "__main__":
    torch.manual_seed(20211122)
    main()
