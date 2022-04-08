#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright    2021  Xiaomi Corp.        (authors: Mingshuang Luo)
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


from pathlib import Path
from typing import Dict, List

import k2
import torch
from tqdm import tqdm

from icefall.lexicon import Lexicon, read_lexicon


class PinyinCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        lexicon: Lexicon,
        device: torch.device,
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
        oov: str = "<unk>",
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang_char/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary token. When a word(token) in the transcript
            does not exist in the token list, it is replaced with `oov`.
        """

        assert oov in lexicon.token_table

        self.lang_dir = lang_dir
        self.oov_id = lexicon.token_table[oov]
        self.token_table = lexicon.token_table

        self.device = device

        self.sos_id = self.token_table[sos_token]
        self.eos_id = self.token_table[eos_token]

        self.word_table = lexicon.word_table
        self.token_table = lexicon.token_table

        self.text2words = convert_text_to_word_segments(
            text_filename=self.lang_dir / "text",
            words_segments_filename=self.lang_dir / "text_words_segmentation",
        )
        self.ragged_lexicon = convert_lexicon_to_ragged(
            filename=self.lang_dir / "lexicon.txt",
            word_table=self.word_table,
            token_table=self.token_table,
        )

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of pinyin-based token IDs.

        Args:
          texts:
            It is a list of strings.
            An example containing two strings is given below:

                ['你好中国', '北京欢迎您']
        Returns:
          Return a list-of-list of pinyin-based token IDs.
        """
        word_ids_list = []
        for i in range(len(texts)):
            word_ids = []
            text = texts[i].strip("\n").strip("\t")
            for word in text.split(" "):
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        ragged_indexes = k2.RaggedTensor(word_ids_list, dtype=torch.int32)
        ans = self.ragged_lexicon.index(ragged_indexes)
        ans = ans.remove_axis(ans.num_axes - 2)

        return ans

    def compile(
        self,
        token_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list token IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
         modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        return k2.ctc_graph(token_ids, modified=modified, device=self.device)


def convert_lexicon_to_ragged(
    filename: str, word_table: k2.SymbolTable, token_table: k2.SymbolTable
) -> k2.RaggedTensor:
    """Read a lexicon and convert lexicon to a ragged tensor.

    Args:
      filename:
        Path to the lexicon file.
      word_table:
        The word symbol table.
      token_table:
        The token symbol table.
    Returns:
      A k2 ragged tensor with two axes [word][token].
    """
    num_words = len(word_table.symbols)
    excluded_words = [
        "<eps>",
        "!SIL",
        "<SPOKEN_NOISE>",
        "<UNK>",
        "#0",
        "<s>",
        "</s>",
    ]

    row_splits = [0]
    token_ids_list = []

    lexicon_tmp = read_lexicon(filename)
    lexicon = dict(lexicon_tmp)
    if len(lexicon_tmp) != len(lexicon):
        raise RuntimeError(
            "It's assumed that each word has a unique pronunciation"
        )

    for i in range(num_words):
        w = word_table[i]
        if w in excluded_words:
            row_splits.append(row_splits[-1])
            continue
        tokens = lexicon[w]
        token_ids = [token_table[k] for k in tokens]

        row_splits.append(row_splits[-1] + len(token_ids))
        token_ids_list.extend(token_ids)

    cached_tot_size = row_splits[-1]
    row_splits = torch.tensor(row_splits, dtype=torch.int32)

    shape = k2.ragged.create_ragged_shape2(
        row_splits,
        None,
        cached_tot_size,
    )
    values = torch.tensor(token_ids_list, dtype=torch.int32)

    return k2.RaggedTensor(shape, values)


def convert_text_to_word_segments(
    text_filename: str, words_segments_filename: str
) -> Dict[str, str]:
    """Convert text to word-based segments.

    Args:
      text_filename:
        The file for the original transcripts.
      words_segments_filename:
        The file after implementing chinese word segmentation
        for the original transcripts.
    Returns:
      A dictionary about text and words_segments.
    """
    text2words = {}

    f_text = open(text_filename, "r", encoding="utf-8")
    text_lines = f_text.readlines()
    text_lines = [line.strip("\t") for line in text_lines]

    f_words = open(words_segments_filename, "r", encoding="utf-8")
    words_lines = f_words.readlines()
    words_lines = [line.strip("\t") for line in words_lines]

    if len(text_lines) != len(words_lines):
        raise RuntimeError(
            "The lengths of text and words_segments should be equal."
        )

    for i in tqdm(range(len(text_lines))):
        text = text_lines[i].strip(" ").strip("\n")
        words_segments = words_lines[i].strip(" ").strip("\n")
        text2words[text] = words_segments

    return text2words
