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


import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import k2
import torch


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
      A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def write_lexicon(filename: str, lexicon: List[Tuple[str, List[str]]]) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")


def convert_lexicon_to_ragged(
    filename: str, word_table: k2.SymbolTable, token_table: k2.SymbolTable
) -> k2.RaggedTensor:
    """Read a lexicon and convert it to a ragged tensor.

    The ragged tensor has two axes: [word][token].

    Caution:
      We assume that each word has a unique pronunciation.

    Args:
      filename:
        Filename of the lexicon. It has a format that can be read
        by :func:`read_lexicon`.
      word_table:
        The word symbol table.
      token_table:
        The token symbol table.
    Returns:
      A k2 ragged tensor with two axes [word][token].
    """
    disambig_id = word_table["#0"]
    # We reuse the same words.txt from the phone based lexicon
    # so that we can share the same G.fst. Here, we have to
    # exclude some words present only in the phone based lexicon.
    excluded_words = ["<eps>", "!SIL", "<SPOKEN_NOISE>"]

    # epsilon is not a word, but it occupies a position
    #
    row_splits = [0]
    token_ids_list = []

    lexicon_tmp = read_lexicon(filename)
    lexicon = dict(lexicon_tmp)
    if len(lexicon_tmp) != len(lexicon):
        raise RuntimeError("It's assumed that each word has a unique pronunciation")

    for i in range(disambig_id):
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


class Lexicon(object):
    """Phone based lexicon."""

    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Args:
          lang_dir:
            Path to the lang directory. It is expected to contain the following
            files:
                - tokens.txt
                - words.txt
                - L.pt
            The above files are produced by the script `prepare.sh`. You
            should have run that before running the training code.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

        if (lang_dir / "Linv.pt").exists():
            logging.info(f"Loading pre-compiled {lang_dir}/Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logging.info("Converting L.pt to Linv.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript FSAs, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.disambig_pattern = disambig_pattern

    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        """
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not self.disambig_pattern.match(s):
                ans.append(self.token_table[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans


class UniqLexicon(Lexicon):
    def __init__(
        self,
        lang_dir: Path,
        uniq_filename: str = "uniq_lexicon.txt",
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Refer to the help information in Lexicon.__init__.

        uniq_filename: It is assumed to be inside the given `lang_dir`.

        Each word in the lexicon is assumed to have a unique pronunciation.
        """
        lang_dir = Path(lang_dir)
        super().__init__(lang_dir=lang_dir, disambig_pattern=disambig_pattern)

        self.ragged_lexicon = convert_lexicon_to_ragged(
            filename=lang_dir / uniq_filename,
            word_table=self.word_table,
            token_table=self.token_table,
        )
        # TODO: should we move it to a certain device ?

    def texts_to_token_ids(
        self, texts: List[str], oov: str = "<UNK>"
    ) -> k2.RaggedTensor:
        """
        Args:
          texts:
            A list of transcripts. Each transcript contains space(s)
            separated words. An example texts is::

                ['HELLO k2', 'HELLO icefall']
          oov:
            The OOV word. If a word in `texts` is not in the lexicon, it is
            replaced with `oov`.
        Returns:
          Return a ragged int tensor with 2 axes [utterance][token_id]
        """
        oov_id = self.word_table[oov]

        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(oov_id)
            word_ids_list.append(word_ids)
        ragged_indexes = k2.RaggedTensor(word_ids_list, dtype=torch.int32)
        ans = self.ragged_lexicon.index(ragged_indexes)
        ans = ans.remove_axis(ans.num_axes - 2)
        return ans

    def words_to_token_ids(self, words: List[str]) -> k2.RaggedTensor:
        """Convert a list of words to a ragged tensor containing token IDs.

        We assume there are no OOVs in "words".
        """
        word_ids = [self.word_table[w] for w in words]
        word_ids = torch.tensor(word_ids, dtype=torch.int32)

        ragged, _ = self.ragged_lexicon.index(
            indexes=word_ids,
            axis=0,
            need_value_indexes=False,
        )
        return ragged
