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

    assert "#0" in tokens

    first_token_disambig_id = tokens.find("#0")

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

    if True:
        logging.info("Determinize star LG")
        kaldifst.determinize_star(LG)

        logging.info("minimizeencoded")
        kaldifst.minimize_encoded(LG)
    else:
        # You can use this branch to compare the size of
        # the resulting graph
        logging.info("Determinize LG")
        LG = kaldifst.determinize(LG)

    LG = k2.Fsa.from_openfst(LG.to_str(is_acceptor=False), acceptor=False)

    LG.labels[LG.labels >= first_token_disambig_id] = 0

    # We only need the labels of LG during beam search decoding
    del LG.aux_labels

    LG = k2.remove_epsilon(LG)
    logging.info(
        f"LG shape after k2.remove_epsilon: {LG.shape}, num_arcs: {LG.num_arcs}"
    )

    LG = k2.connect(LG)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    return LG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    out_filename = lang_dir / "LG.pt"

    if out_filename.is_file():
        logging.info(f"{out_filename} already exists - skipping")
        return

    logging.info(f"Processing {lang_dir}")

    LG = compile_LG(lang_dir)
    logging.info(f"Saving LG to {out_filename}")
    torch.save(LG.as_dict(), out_filename)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
