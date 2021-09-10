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


# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

"""

This script takes as input `lang_dir`, which should contain::

    - lang_dir/bpe.model,
    - lang_dir/words.txt

and generates the following files in the directory `lang_dir`:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - tokens.txt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import k2
import sentencepiece as spm
import torch
from prepare_lang import (
    Lexicon,
    add_disambig_symbols,
    lexicon_to_fst,
    write_lexicon,
    write_mapping,
)


def generate_lexicon(
    model_file: str, words: List[str]
) -> Tuple[Lexicon, Dict[str, int]]:
    """Generate a lexicon from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      words:
        A list of strings representing words.
    Returns:
      Return a tuple with two elements:
        - A dict whose keys are words and values are the corresponding
          word pieces.
        - A dict representing the token symbol, mapping from tokens to IDs.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    words_pieces: List[List[str]] = sp.encode(words, out_type=str)

    lexicon = []
    for word, pieces in zip(words, words_pieces):
        lexicon.append((word, pieces))

    # The OOV word is <UNK>
    lexicon.append(("<UNK>", [sp.id_to_piece(sp.unk_id())]))

    token2id: Dict[str, int] = dict()
    for i in range(sp.vocab_size()):
        token2id[sp.id_to_piece(i)] = i

    return lexicon, token2id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain the bpe.model and words.txt
        """,
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    model_file = lang_dir / "bpe.model"

    word_sym_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    words = word_sym_table.symbols

    excluded = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]
    for w in excluded:
        if w in words:
            words.remove(w)

    lexicon, token_sym_table = generate_lexicon(model_file, words)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    next_token_id = max(token_sym_table.values()) + 1
    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in token_sym_table
        token_sym_table[disambig] = next_token_id
        next_token_id += 1

    word_sym_table.add("#0")
    word_sym_table.add("<s>")
    word_sym_table.add("</s>")

    write_mapping(lang_dir / "tokens.txt", token_sym_table)

    write_lexicon(lang_dir / "lexicon.txt", lexicon)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst(
        lexicon,
        token2id=token_sym_table,
        word2id=word_sym_table,
    )

    L_disambig = lexicon_to_fst(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word_sym_table,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")


if __name__ == "__main__":
    main()
