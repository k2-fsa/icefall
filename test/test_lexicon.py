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


from pathlib import Path

import k2
import pytest
import torch

from icefall.lexicon import BpeLexicon, Lexicon


@pytest.fixture
def lang_dir(tmp_path):
    phone2id = """
        <eps> 0
        a 1
        b 2
        f 3
        o 4
        r 5
        z 6
        SPN 7
        #0 8
    """
    word2id = """
        <eps> 0
        foo 1
        bar 2
        baz 3
        <UNK> 4
        #0 5
    """

    L = k2.Fsa.from_str(
        """
        0 0 7 4 0
        0 7 -1 -1 0
        0 1 3 1 0
        0 3 2 2 0
        0 5 2 3 0
        1 2 4 0 0
        2 0 4 0 0
        3 4 1 0 0
        4 0 5 0 0
        5 6 1 0 0
        6 0 6 0 0
        7
    """,
        num_aux_labels=1,
    )

    with open(tmp_path / "tokens.txt", "w") as f:
        f.write(phone2id)
    with open(tmp_path / "words.txt", "w") as f:
        f.write(word2id)

    torch.save(L.as_dict(), tmp_path / "L.pt")

    return tmp_path


def test_lexicon(lang_dir):
    lexicon = Lexicon(lang_dir)
    assert lexicon.tokens == list(range(1, 8))


def test_bpe_lexicon():
    lang_dir = Path("data/lang/bpe")
    if not lang_dir.is_dir():
        return
    # TODO: Generate test data for BpeLexicon

    lexicon = BpeLexicon(lang_dir)
    words = ["<UNK>", "HELLO", "ZZZZ", "WORLD"]
    ids = lexicon.words_to_piece_ids(words)
    print(ids)
    print([lexicon.token_table[i] for i in ids.values().tolist()])
