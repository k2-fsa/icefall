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
import pytest
import torch

from icefall.utils import AttributeDict, encode_supervisions, get_texts


@pytest.fixture
def sup():
    sequence_idx = torch.tensor([0, 1, 2])
    start_frame = torch.tensor([1, 3, 9])
    num_frames = torch.tensor([20, 30, 10])
    text = ["one", "two", "three"]
    return {
        "sequence_idx": sequence_idx,
        "start_frame": start_frame,
        "num_frames": num_frames,
        "text": text,
    }


def test_encode_supervisions(sup):
    supervision_segments, texts = encode_supervisions(sup, subsampling_factor=4)
    assert torch.all(
        torch.eq(
            supervision_segments,
            torch.tensor(
                [[1, 0, 30 // 4], [0, 0, 20 // 4], [2, 9 // 4, 10 // 4]]
            ),
        )
    )
    assert texts == ["two", "one", "three"]


def test_get_texts_ragged():
    fsa1 = k2.Fsa.from_str(
        """
        0 1 1 10
        1 2 2 20
        2 3 3 30
        3 4 -1 0
        4
    """
    )
    fsa1.aux_labels = k2.RaggedTensor("[ [1 3 0 2] [] [4 0 1] [-1]]")

    fsa2 = k2.Fsa.from_str(
        """
        0 1 1 1
        1 2 2 2
        2 3 -1 0
        3
    """
    )
    fsa2.aux_labels = k2.RaggedTensor("[[3 0 5 0 8] [0 9 7 0] [-1]]")
    fsas = k2.Fsa.from_fsas([fsa1, fsa2])
    texts = get_texts(fsas)
    assert texts == [[1, 3, 2, 4, 1], [3, 5, 8, 9, 7]]


def test_get_texts_regular():
    fsa1 = k2.Fsa.from_str(
        """
        0 1 1 3 10
        1 2 2 0 20
        2 3 3 2 30
        3 4 -1 -1 0
        4
    """,
        num_aux_labels=1,
    )

    fsa2 = k2.Fsa.from_str(
        """
        0 1 1 10 1
        1 2 2 5 2
        2 3 -1 -1 0
        3
    """,
        num_aux_labels=1,
    )
    fsas = k2.Fsa.from_fsas([fsa1, fsa2])
    texts = get_texts(fsas)
    assert texts == [[3, 2], [10, 5]]


def test_attribute_dict():
    s = AttributeDict({"a": 10, "b": 20})
    assert s.a == 10
    assert s["b"] == 20
    s.c = 100
    assert s["c"] == 100
    assert hasattr(s, "a")
    assert hasattr(s, "b")
    assert getattr(s, "a") == 10
    del s.a
    assert hasattr(s, "a") is False
    setattr(s, "c", 100)
    s.c = 100
    try:
        del s.a
    except AttributeError as ex:
        print(f"Caught exception: {ex}")
