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


"""
This script takes as input lang_dir and generates LG from

    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated LG is saved in $lang_dir/LG.fst
"""

import argparse
import logging
from pathlib import Path

import k2
import kaldifst
import torch
from kaldifst.utils import k2_to_openfst


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    return parser.parse_args()


def compile_LG(lang_dir: str) -> kaldifst.StdVectorFst:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_500.

    Return:
      An FST representing LG.
    """

    tokens = kaldifst.SymbolTable.read_text(f"{lang_dir}/tokens.txt")
    words = kaldifst.SymbolTable.read_text(f"{lang_dir}/words.txt")

    assert "#0" in tokens
    assert "#0" in words

    token_disambig_id = tokens.find("#0")
    word_disambig_id = words.find("#0")

    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))
    L = k2_to_openfst(L, olabels="aux_labels")

    kaldifst.arcsort(L, sort_type="olabel")
    L.write(f"{lang_dir}/L.fst")

    with open("data/lm/G_3_gram.fst.txt") as f:
        G = kaldifst.compile(
            f.read(),
            acceptor=False,
            fst_type="vector",
            arc_type="standard",
        )
    kaldifst.arcsort(G, sort_type="ilabel")

    logging.info("Composing LG")
    LG = kaldifst.compose(
        L,
        G,
        match_side="left",
        compose_filter="sequence",
        connect=True,
    )

    logging.info("Determinize star LG")
    kaldifst.determinize_star(LG)

    logging.info("minimizeencoded")
    kaldifst.minimize_encoded(LG)

    # Set all disambig IDs to eps
    for state in kaldifst.StateIterator(LG):
        for arc in kaldifst.ArcIterator(LG, state):
            if arc.ilabel >= token_disambig_id:
                arc.ilabel = 0

            if arc.olabel >= word_disambig_id:
                arc.olabel = 0
    # reset properties as we changed the arc labels above
    LG.properties(0xFFFFFFFF, True)

    return LG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    out_filename = lang_dir / "LG.fst"

    if out_filename.is_file():
        logging.info(f"{out_filename} already exists - skipping")
        return

    logging.info(f"Processing {lang_dir}")

    LG = compile_LG(lang_dir)
    logging.info(f"Saving LG to {out_filename}")
    LG.write(str(out_filename))


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
