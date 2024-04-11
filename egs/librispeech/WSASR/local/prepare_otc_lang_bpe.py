#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2023  Johns Hopkins University (author: Dongji Gao)
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
    add_self_loops,
    write_lexicon,
    write_mapping,
)

from icefall.utils import str2bool


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # The blank symbol <blk> is defined in local/train_bpe_model.py
    assert token2id["<blk>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def generate_otc_lexicon(
    model_file: str,
    words: List[str],
    oov: str,
    otc_token: str,
) -> Tuple[Lexicon, Dict[str, int]]:
    """Generate a lexicon from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      words:
        A list of strings representing words.
      oov:
        The out of vocabulary word in lexicon.
      otc_token:
        The OTC token in lexicon.
    Returns:
      Return a tuple with two elements:
        - A dict whose keys are words and values are the corresponding
          word pieces.
        - A dict representing the token symbol, mapping from tokens to IDs.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    # Convert word to word piece IDs instead of word piece strings
    # to avoid OOV tokens.
    words_pieces_ids: List[List[int]] = sp.encode(words, out_type=int)

    # Now convert word piece IDs back to word piece strings.
    words_pieces: List[List[str]] = [sp.id_to_piece(ids) for ids in words_pieces_ids]

    lexicon = []
    for word, pieces in zip(words, words_pieces):
        lexicon.append((word, pieces))

    lexicon.append((oov, ["▁", sp.id_to_piece(sp.unk_id())]))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}

    # Add OTC token to the last.
    lexicon.append((otc_token, [f"▁{otc_token}"]))
    otc_token_index = len(token2id)
    token2id[f"▁{otc_token}"] = otc_token_index

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

    parser.add_argument(
        "--oov",
        type=str,
        default="<UNK>",
        help="The out of vocabulary word in lexicon.",
    )

    parser.add_argument(
        "--otc-token",
        type=str,
        default="<star>",
        help="The OTC token in lexicon.",
    )

    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="""True for debugging, which will generate
        a visualization of the lexicon FST.

        Caution: If your lexicon contains hundreds of thousands
        of lines, please set it to False!

        See "test/test_bpe_lexicon.py" for usage.
        """,
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    model_file = lang_dir / "bpe.model"
    otc_token = args.otc_token

    word_sym_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

    words = word_sym_table.symbols

    excluded = [
        "<eps>",
        "!SIL",
        "<SPOKEN_NOISE>",
        args.oov,
        otc_token,
        "#0",
        "<s>",
        "</s>",
    ]

    for w in excluded:
        if w in words:
            words.remove(w)

    lexicon, token_sym_table = generate_otc_lexicon(
        model_file, words, args.oov, otc_token
    )

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

    L = lexicon_to_fst_no_sil(
        lexicon,
        token2id=token_sym_table,
        word2id=word_sym_table,
    )

    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token_sym_table,
        word2id=word_sym_table,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")

    if args.debug:
        labels_sym = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        aux_labels_sym = k2.SymbolTable.from_file(lang_dir / "words.txt")

        L.labels_sym = labels_sym
        L.aux_labels_sym = aux_labels_sym
        L.draw(f"{lang_dir / 'L.svg'}", title="L.pt")

        L_disambig.labels_sym = labels_sym
        L_disambig.aux_labels_sym = aux_labels_sym
        L_disambig.draw(f"{lang_dir / 'L_disambig.svg'}", title="L_disambig.pt")


if __name__ == "__main__":
    main()
