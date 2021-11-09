#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
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
import k2
import torch

from icefall.lm.rescore import (
    add_bos,
    add_eos,
    compute_alignment,
    make_hyp_to_ref_map,
    make_repeat,
    make_repeat_map,
    prepare_conformer_lm_inputs,
)


def test_add_bos():
    bos_id = 100
    ragged = k2.RaggedTensor([[1, 2], [3], [0]])
    bos_ragged = add_bos(ragged, bos_id)
    expected = k2.RaggedTensor([[bos_id, 1, 2], [bos_id, 3], [bos_id, 0]])
    assert str(bos_ragged) == str(expected)


def test_add_eos():
    eos_id = 30
    ragged = k2.RaggedTensor([[1, 2], [3], [], [5, 8, 9]])
    ragged_eos = add_eos(ragged, eos_id)
    expected = k2.RaggedTensor(
        [[1, 2, eos_id], [3, eos_id], [eos_id], [5, 8, 9, eos_id]]
    )
    assert str(ragged_eos) == str(expected)


def test_pad():
    bos_id = 10
    eos_id = 100
    ragged = k2.RaggedTensor([[1, 2, 3], [5], [9, 8]])
    bos_ragged = add_bos(ragged, bos_id)
    bos_ragged_eos = add_eos(bos_ragged, eos_id)
    blank_id = -1
    padded = bos_ragged_eos.pad(mode="constant", padding_value=blank_id)
    expected = torch.tensor(
        [
            [bos_id, 1, 2, 3, eos_id],
            [bos_id, 5, eos_id, blank_id, blank_id],
            [bos_id, 9, 8, eos_id, blank_id],
        ]
    ).to(padded)
    assert torch.all(torch.eq(padded, expected))


def test_make_hyp_to_ref_map():
    a = k2.RaggedTensor([[[1, 2], [], [3]], [[1, 3], [2], [4], [5]]])
    row_splits = a.shape.row_splits(1)
    repeat_map = make_hyp_to_ref_map(row_splits)
    # fmt: off
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
        3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6]).to(repeat_map)  # noqa
    # fmt: on
    assert torch.all(torch.eq(repeat_map, expected))


def test_make_repeat_map():
    a = k2.RaggedTensor([[[1, 2], [], [3]], [[1, 3], [2], [4], [5]]])
    row_splits = a.shape.row_splits(1)
    repeat_map = make_repeat_map(row_splits)
    # fmt: off
    expected = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2,
        3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6,  # noqa
        3, 4, 5, 6]).to(repeat_map)  # noqa
    # fmt: on
    assert torch.all(torch.eq(repeat_map, expected))


def test_make_repeat():
    # fmt: off
    a = k2.RaggedTensor([
        [[1, 3, 5], [2, 6]],
        [[1, 2, 3, 4], [2], [], [9, 10, 11]],
        ])
    b = make_repeat(a)
    expected = k2.RaggedTensor([
        [[1, 3, 5], [2, 6], [1, 3, 5], [2, 6]],
        [[1, 2, 3, 4], [2], [], [9, 10, 11],
         [1, 2, 3, 4], [2], [], [9, 10, 11],
         [1, 2, 3, 4], [2], [], [9, 10, 11],
         [1, 2, 3, 4], [2], [], [9, 10, 11]],
        ])
    # fmt: on
    assert str(b) == str(expected)


def test_compute_alignment():
    # fmt: off
    tokens = k2.RaggedTensor([
        # utt 0
        [1, 3, 5, 8], [1,  5, 8], [2, 8, 3, 2],
        # utt 1
        [2, 3], [2],
        ])
    # fmt: on
    shape = k2.RaggedShape("[[x x x] [x x]]")
    alignment = compute_alignment(tokens, shape)
    (
        masked_src,
        src,
        tgt,
        src_key_padding_mask,
        weight,
    ) = prepare_conformer_lm_inputs(alignment, bos_id=10, eos_id=20, blank_id=0)

    #  print("masked src", masked_src)
    #  print("src", src)
    #  print("tgt", tgt)
    #  print("src_key_padding_mask", src_key_padding_mask)
    #  print("weight", weight)


def main():
    test_add_bos()
    test_add_eos()
    test_pad()
    test_make_repeat_map()
    test_make_hyp_to_ref_map()
    test_make_repeat()
    test_compute_alignment()


if __name__ == "__main__":
    main()
