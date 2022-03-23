#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    return parser.parse_args()


def compile_LG(lang_dir: str) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_500.

    Return:
      An FST representing LG.
    """

    tokens = k2.SymbolTable.from_file(f"{lang_dir}/tokens.txt")

    assert "#0" in tokens

    first_token_disambig_id = tokens["#0"]
    logging.info(f"first token disambig ID: {first_token_disambig_id}")

    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path("data/lm/G_3_gram.pt").is_file():
        logging.info("Loading pre-compiled G_3_gram")
        d = torch.load("data/lm/G_3_gram.pt")
        G = k2.Fsa.from_dict(d)
    else:
        logging.info("Loading G_3_gram.fst.txt")
        with open("data/lm/G_3_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            del G.aux_labels
            torch.save(G.as_dict(), "data/lm/G_3_gram.pt")

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Composing L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}, num_arcs: {LG.num_arcs}")

    del LG.aux_labels

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(
        f"LG shape after k2.connect: {LG.shape}, num_arcs: {LG.num_arcs}"
    )

    logging.info("Determinizing LG")
    LG = k2.determinize(LG)
    logging.info(
        f"LG shape after k2.determinize: {LG.shape}, num_arcs: {LG.num_arcs}"
    )

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)
    logging.info(
        f"LG shape after k2.connect: {LG.shape}, num_arcs: {LG.num_arcs}"
    )

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info(f"LG properties: {LG.properties_str}")
    # Possible properties is:
    # "Valid|Nonempty|ArcSorted|ArcSortedAndDeterministic|EpsilonFree|MaybeAccessible|MaybeCoaccessible"  # noqa
    logging.info(
        "Caution: LG is deterministic and contains disambig symbols!!!"
    )

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
