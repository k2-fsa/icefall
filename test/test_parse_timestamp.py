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

    text = "HELLO WORLD"
    token_ids = sp.encode(text, out_type=int)
    # out_type=str: ['_HE', 'LL', 'O', '_WORLD']
    # out_type=int: [22, 58, 24, 425]

    # [22, 22, 58, 24, 0, 0, 425, 425, 425, 0, 0]
    labels = (
        token_ids[0:1] * 2 + token_ids[1:3] + [0] * 2 + token_ids[3:4] * 3 + [0] * 2
    )
    # [22, 0, 58, 24, 0, 0, 425, 0, 0, 0, 0]
    aux_labels = (
        token_ids[0:1]
        + [0]
        + token_ids[1:3]
        + [0] * 2
        + token_ids[3:4]
        + [0] * 4
        + [-1]
    )

    fsa = k2.linear_fsa(labels)
    fsa.aux_labels = torch.tensor(aux_labels).to(torch.int32)

    fsa_vec = k2.create_fsa_vec([fsa])

    utt_index_pairs, utt_words = parse_bpe_timestamps_and_texts(fsa_vec, sp)
    assert utt_index_pairs[0] == [(0, 3), (6, 8)], utt_index_pairs[0]
    assert utt_words[0] == ["HELLO", "WORLD"], utt_words[0]


def test_parse_timestamps_and_texts():
    lang_dir = ICEFALL_DIR / "egs/librispeech/ASR/data/lang_bpe_500"
    if not lang_dir.is_dir():
        print(f"{lang_dir} does not exist.")
        return

    lexicon = Lexicon(lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))

    text = "HELLO WORLD"
    token_ids = sp.encode(text, out_type=int)
    # out_type=str: ['_HE', 'LL', 'O', '_WORLD']
    # out_type=int: [22, 58, 24, 425]

    word_table = lexicon.word_table
    word_ids = [word_table[s] for s in text.split()]  # [79677, 196937]

    # [22, 22, 58, 24, 0, 0, 425, 425, 425, 0, 0]
    labels = (
        token_ids[0:1] * 2 + token_ids[1:3] + [0] * 2 + token_ids[3:4] * 3 + [0] * 2
    )

    # [[79677], [], [], [], [], [], [196937], [], [], [], [], []]
    aux_labels = [word_ids[0:1]] + [[]] * 5 + [word_ids[1:2]] + [[]] * 5

    fsa = k2.linear_fsa(labels)
    fsa.aux_labels = k2.RaggedTensor(aux_labels)

    fsa_vec = k2.create_fsa_vec([fsa, fsa])

    utt_index_pairs, utt_words = parse_timestamps_and_texts(fsa_vec, word_table)
    assert utt_index_pairs[0] == [(0, 3), (6, 8)], utt_index_pairs[0]
    assert utt_words[0] == ["HELLO", "WORLD"], utt_words[0]


if __name__ == "__main__":
    test_parse_bpe_timestamps_and_texts()
    test_parse_timestamps_and_texts()
