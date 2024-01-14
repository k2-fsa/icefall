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


import torch
from torch.nn.utils.rnn import pad_sequence
from transformer import (
    Transformer,
    add_eos,
    add_sos,
    decoder_padding_mask,
    encoder_padding_mask,
    generate_square_subsequent_mask,
)


def test_encoder_padding_mask():
    supervisions = {
        "sequence_idx": torch.tensor([0, 1, 2]),
        "start_frame": torch.tensor([0, 0, 0]),
        "num_frames": torch.tensor([18, 7, 13]),
    }

    max_len = ((18 - 1) // 2 - 1) // 2
    mask = encoder_padding_mask(max_len, supervisions)
    expected_mask = torch.tensor(
        [
            [False, False, False],  # ((18 - 1)//2 - 1)//2 = 3,
            [False, True, True],  # ((7 - 1)//2 - 1)//2 = 1,
            [False, False, True],  # ((13 - 1)//2 - 1)//2 = 2,
        ]
    )
    assert torch.all(torch.eq(mask, expected_mask))


def test_transformer():
    num_features = 40
    num_classes = 87
    model = Transformer(num_features=num_features, num_classes=num_classes)

    N = 31

    for T in range(7, 30):
        x = torch.rand(N, T, num_features)
        y, _, _ = model(x)
        assert y.shape == (N, (((T - 1) // 2) - 1) // 2, num_classes)


def test_generate_square_subsequent_mask():
    s = 5
    mask = generate_square_subsequent_mask(s)
    inf = float("inf")
    expected_mask = torch.tensor(
        [
            [0.0, -inf, -inf, -inf, -inf],
            [0.0, 0.0, -inf, -inf, -inf],
            [0.0, 0.0, 0.0, -inf, -inf],
            [0.0, 0.0, 0.0, 0.0, -inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert torch.all(torch.eq(mask, expected_mask))


def test_decoder_padding_mask():
    x = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([2, 5, 8])]
    y = pad_sequence(x, batch_first=True, padding_value=-1)
    mask = decoder_padding_mask(y, ignore_id=-1)
    expected_mask = torch.tensor(
        [[False, False, True], [False, True, True], [False, False, False]]
    )
    assert torch.all(torch.eq(mask, expected_mask))


def test_add_sos():
    x = [[1, 2], [3], [2, 5, 8]]
    y = add_sos(x, sos_id=0)
    expected_y = [[0, 1, 2], [0, 3], [0, 2, 5, 8]]
    assert y == expected_y


def test_add_eos():
    x = [[1, 2], [3], [2, 5, 8]]
    y = add_eos(x, eos_id=0)
    expected_y = [[1, 2, 0], [3, 0], [2, 5, 8, 0]]
    assert y == expected_y
