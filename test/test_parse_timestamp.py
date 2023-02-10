#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
import sentencepiece as spm
import torch

from icefall.lexicon import Lexicon
from icefall.utils import parse_bpe_timestamps_and_texts, parse_timestamps_and_texts

ICEFALL_DIR = Path(__file__).resolve().parent.parent


def test_parse_bpe_timestamps_and_texts():
    lang_dir = ICEFALL_DIR / "egs/librispeech/ASR/data/lang_bpe_500"
    if not lang_dir.is_dir():
        print(f"{lang_dir} does not exist.")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))

    text_1 = "HELLO WORLD"
    token_ids_1 = sp.encode(text_1, out_type=int)
    # out_type=str: ['_HE', 'LL', 'O', '_WORLD']
    # out_type=int: [22, 58, 24, 425]

    # [22, 22, 58, 24, 0, 0, 425, 425, 425, 0, 0]
    labels_1 = (
        token_ids_1[0:1] * 2
        + token_ids_1[1:3]
        + [0] * 2
        + token_ids_1[3:4] * 3
        + [0] * 2
    )
    # [22, 0, 58, 24, 0, 0, 425, 0, 0, 0, 0, -1]
    aux_labels_1 = (
        token_ids_1[0:1]
        + [0]
        + token_ids_1[1:3]
        + [0] * 2
        + token_ids_1[3:4]
        + [0] * 4
        + [-1]
    )
    fsa_1 = k2.linear_fsa(labels_1)
    fsa_1.aux_labels = torch.tensor(aux_labels_1).to(torch.int32)

    text_2 = "SAY GOODBYE"
    token_ids_2 = sp.encode(text_2, out_type=int)
    # out_type=str: ['_SAY', '_GOOD', 'B', 'Y', 'E']
    # out_type=int: [289, 286, 41, 16, 11]

    # [289, 0, 0, 286, 286, 41, 16, 11, 0, 0]
    labels_2 = (
        token_ids_2[0:1] + [0] * 2 + token_ids_2[1:2] * 2 + token_ids_2[2:5] + [0] * 2
    )
    # [289, 0, 0, 286, 0, 41, 16, 11, 0, 0, -1]
    aux_labels_2 = (
        token_ids_2[0:1]
        + [0] * 2
        + token_ids_2[1:2]
        + [0]
        + token_ids_2[2:5]
        + [0] * 2
        + [-1]
    )
    fsa_2 = k2.linear_fsa(labels_2)
    fsa_2.aux_labels = torch.tensor(aux_labels_2).to(torch.int32)

    fsa_vec = k2.create_fsa_vec([fsa_1, fsa_2])

    utt_index_pairs, utt_words = parse_bpe_timestamps_and_texts(fsa_vec, sp)
    assert utt_index_pairs[0] == [(0, 3), (6, 8)], utt_index_pairs[0]
    assert utt_words[0] == ["HELLO", "WORLD"], utt_words[0]
    assert utt_index_pairs[1] == [(0, 0), (3, 7)], utt_index_pairs[1]
    assert utt_words[1] == ["SAY", "GOODBYE"], utt_words[1]


def test_parse_timestamps_and_texts():
    lang_dir = ICEFALL_DIR / "egs/librispeech/ASR/data/lang_bpe_500"
    if not lang_dir.is_dir():
        print(f"{lang_dir} does not exist.")
        return

    lexicon = Lexicon(lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    word_table = lexicon.word_table

    text_1 = "HELLO WORLD"
    token_ids_1 = sp.encode(text_1, out_type=int)
    # out_type=str: ['_HE', 'LL', 'O', '_WORLD']
    # out_type=int: [22, 58, 24, 425]
    word_ids_1 = [word_table[s] for s in text_1.split()]  # [79677, 196937]
    # [22, 22, 58, 24, 0, 0, 425, 425, 425, 0, 0]
    labels_1 = (
        token_ids_1[0:1] * 2
        + token_ids_1[1:3]
        + [0] * 2
        + token_ids_1[3:4] * 3
        + [0] * 2
    )
    # [[79677], [], [], [], [], [], [196937], [], [], [], [], []]
    aux_labels_1 = [word_ids_1[0:1]] + [[]] * 5 + [word_ids_1[1:2]] + [[]] * 5

    fsa_1 = k2.linear_fsa(labels_1)
    fsa_1.aux_labels = k2.RaggedTensor(aux_labels_1)

    text_2 = "SAY GOODBYE"
    token_ids_2 = sp.encode(text_2, out_type=int)
    # out_type=str: ['_SAY', '_GOOD', 'B', 'Y', 'E']
    # out_type=int: [289, 286, 41, 16, 11]
    word_ids_2 = [word_table[s] for s in text_2.split()]  # [154967, 72079]
    # [289, 0, 0, 286, 286, 41, 16, 11, 0, 0]
    labels_2 = (
        token_ids_2[0:1] + [0] * 2 + token_ids_2[1:2] * 2 + token_ids_2[2:5] + [0] * 2
    )
    # [[154967], [], [], [72079], [], [], [], [], [], [], []]
    aux_labels_2 = [word_ids_2[0:1]] + [[]] * 2 + [word_ids_2[1:2]] + [[]] * 7

    fsa_2 = k2.linear_fsa(labels_2)
    fsa_2.aux_labels = k2.RaggedTensor(aux_labels_2)

    fsa_vec = k2.create_fsa_vec([fsa_1, fsa_2])

    utt_index_pairs, utt_words = parse_timestamps_and_texts(fsa_vec, word_table)
    assert utt_index_pairs[0] == [(0, 3), (6, 8)], utt_index_pairs[0]
    assert utt_words[0] == ["HELLO", "WORLD"], utt_words[0]
    assert utt_index_pairs[1] == [(0, 0), (3, 7)], utt_index_pairs[1]
    assert utt_words[1] == ["SAY", "GOODBYE"], utt_words[1]


if __name__ == "__main__":
    test_parse_bpe_timestamps_and_texts()
    test_parse_timestamps_and_texts()
