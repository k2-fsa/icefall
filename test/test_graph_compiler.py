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


import re

import k2
import pytest
import torch

from icefall.graph_compiler import CtcTrainingGraphCompiler
from icefall.lexicon import Lexicon
from icefall.utils import get_texts


@pytest.fixture
def lexicon():
    """
    We use the following test data:

    lexicon.txt

        foo f o o
        bar b a r
        baz b a z
        <UNK> SPN

    phones.txt

        <eps> 0
        a 1
        b 2
        f 3
        o 4
        r 5
        z 6
        SPN 7

    words.txt:

        <eps> 0
        foo 1
        bar 2
        baz 3
        <UNK> 4
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
    L.labels_sym = k2.SymbolTable.from_str(
        """
        a 1
        b 2
        f 3
        o 4
        r 5
        z 6
        SPN 7
    """
    )
    L.aux_labels_sym = k2.SymbolTable.from_str(
        """
        foo 1
        bar 2
        baz 3
        <UNK> 4
    """
    )
    ans = Lexicon.__new__(Lexicon)
    ans.token_table = L.labels_sym
    ans.word_table = L.aux_labels_sym
    ans.L_inv = k2.arc_sort(L.invert_())
    ans.disambig_pattern = re.compile(r"^#\d+$")

    return ans


@pytest.fixture
def compiler(lexicon):
    return CtcTrainingGraphCompiler(lexicon, device=torch.device("cpu"))


class TestCtcTrainingGraphCompiler(object):
    @staticmethod
    def test_convert_transcript_to_fsa(compiler, lexicon):
        texts = ["bar foo", "baz ok"]
        fsa = compiler.convert_transcript_to_fsa(texts)
        labels0 = fsa[0].labels[:-1].tolist()
        aux_labels0 = fsa[0].aux_labels[:-1]
        aux_labels0 = aux_labels0[aux_labels0 != 0].tolist()

        labels1 = fsa[1].labels[:-1].tolist()
        aux_labels1 = fsa[1].aux_labels[:-1]
        aux_labels1 = aux_labels1[aux_labels1 != 0].tolist()

        labels0 = [lexicon.token_table[i] for i in labels0]
        labels1 = [lexicon.token_table[i] for i in labels1]

        aux_labels0 = [lexicon.word_table[i] for i in aux_labels0]
        aux_labels1 = [lexicon.word_table[i] for i in aux_labels1]

        assert labels0 == ["b", "a", "r", "f", "o", "o"]
        assert aux_labels0 == ["bar", "foo"]

        assert labels1 == ["b", "a", "z", "SPN"]
        assert aux_labels1 == ["baz", "<UNK>"]

    @staticmethod
    def test_compile(compiler, lexicon):
        texts = ["bar foo", "baz ok"]
        decoding_graph = compiler.compile(texts)
        input1 = ["b", "b", "<blk>", "<blk>", "a", "a", "r", "<blk>", "<blk>"]
        input1 += ["f", "f", "<blk>", "<blk>", "o", "o", "<blk>", "o", "o"]

        input2 = ["b", "b", "a", "a", "a", "<blk>", "<blk>", "z", "z"]
        input2 += ["<blk>", "<blk>", "SPN", "SPN", "<blk>", "<blk>"]

        lexicon.token_table._id2sym[0] == "<blk>"
        lexicon.token_table._sym2id["<blk>"] = 0

        input1 = [lexicon.token_table[i] for i in input1]
        input2 = [lexicon.token_table[i] for i in input2]

        fsa1 = k2.linear_fsa(input1)
        fsa2 = k2.linear_fsa(input2)
        fsas = k2.Fsa.from_fsas([fsa1, fsa2])

        decoding_graph = k2.arc_sort(decoding_graph)
        lattice = k2.intersect(decoding_graph, fsas, treat_epsilons_specially=False)
        lattice = k2.connect(lattice)

        aux_labels0 = lattice[0].aux_labels[:-1]
        aux_labels0 = aux_labels0[aux_labels0 != 0].tolist()
        aux_labels0 = [lexicon.word_table[i] for i in aux_labels0]
        assert aux_labels0 == ["bar", "foo"]

        aux_labels1 = lattice[1].aux_labels[:-1]
        aux_labels1 = aux_labels1[aux_labels1 != 0].tolist()
        aux_labels1 = [lexicon.word_table[i] for i in aux_labels1]
        assert aux_labels1 == ["baz", "<UNK>"]

        texts = get_texts(lattice)
        texts = [[lexicon.word_table[i] for i in words] for words in texts]
        assert texts == [["bar", "foo"], ["baz", "<UNK>"]]
