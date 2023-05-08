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
This script takes as input lang_dir and generates HLG from

    - H, the ctc topology, built from tokens contained in lang_dir/lexicon.txt
    - L, the lexicon, built from lang_dir/L_disambig.fst

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_n_gram.fst.txt

The generated HLG is saved in $lang_dir/HLG_fst.pt

So when to use this script instead of ./local/compile_hlg.py ?
If you have a very large G, ./local/compile_hlg.py may throw OOM for
determinization. In that case, you can use this script to compile HLG.
"""

import argparse
import logging
from pathlib import Path

import k2
import kaldifst
import torch

from icefall.lexicon import Lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lm",
        type=str,
        default="G_3_gram",
        help="""Stem name for LM used in HLG compiling.
        """,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    return parser.parse_args()


def compile_HLG(lang_dir: str, lm: str = "G_3_gram") -> kaldifst.StdVectorFst:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang_phone or data/lang_bpe_5000.
      lm:
        The language stem base name.

    Return:
      An FST representing HLG.
    """

    L = kaldifst.StdVectorFst.read(f"{lang_dir}/L_disambig.fst")
    logging.info("Arc sort L")
    kaldifst.arcsort(L, sort_type="olabel")
    logging.info(f"L: #states {L.num_states}")

    G_filename_txt = f"data/lm/{lm}.fst.txt"
    G_filename_binary = f"data/lm/{lm}.fst"
    if Path(G_filename_binary).is_file():
        logging.info(f"Loading {G_filename_binary}")
        G = kaldifst.StdVectorFst.read(G_filename_binary)
    else:
        logging.info(f"Loading {G_filename_txt}")
        with open(G_filename_txt) as f:
            G = kaldifst.compile(s=f.read(), acceptor=False)
            logging.info(f"Saving G to {G_filename_binary}")
            G.write(G_filename_binary)

    logging.info("Arc sort G")
    kaldifst.arcsort(G, sort_type="ilabel")

    logging.info(f"G: #states {G.num_states}")

    logging.info("Compose L and G and connect LG")
    LG = kaldifst.compose(L, G, connect=True)
    logging.info(f"LG: #states {LG.num_states}")

    logging.info("Determinizestar LG")
    kaldifst.determinize_star(LG)
    logging.info(f"LG after determinize_star: #states {LG.num_states}")

    logging.info("Minimize encoded LG")
    kaldifst.minimize_encoded(LG)
    logging.info(f"LG after minimize_encoded: #states {LG.num_states}")

    logging.info("Converting LG to k2 format")
    LG = k2.Fsa.from_openfst(LG.to_str(is_acceptor=False), acceptor=False)
    logging.info(f"LG in k2: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}")

    lexicon = Lexicon(lang_dir)

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]
    logging.info(f"token id for #0: {first_token_disambig_id}")
    logging.info(f"word id for #0: {first_word_disambig_id}")

    max_token_id = max(lexicon.tokens)
    modified = False
    logging.info(
        f"Building ctc_topo. modified: {modified}, max_token_id: {max_token_id}"
    )

    H = k2.ctc_topo(max_token_id, modified=modified)
    logging.info(f"H: #states: {H.shape[0]}, #arcs: {H.num_arcs}")

    logging.info("Removing disambiguation symbols on LG")
    LG.labels[LG.labels >= first_token_disambig_id] = 0
    LG.aux_labels[LG.aux_labels >= first_word_disambig_id] = 0

    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    logging.info("Removing epsilons from LG")
    LG = k2.remove_epsilon(LG)
    logging.info(
        f"LG after k2.remove_epsilon: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}"
    )

    logging.info("Connecting LG after removing epsilons")
    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)
    logging.info(f"LG after k2.connect: #states: {LG.shape[0]}, #arcs: {LG.num_arcs}")

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing H and LG")

    HLG = k2.compose(H, LG, inner_labels="tokens")
    logging.info(
        f"HLG after k2.compose: #states: {HLG.shape[0]}, #arcs: {HLG.num_arcs}"
    )

    logging.info("Connecting HLG")
    HLG = k2.connect(HLG)
    logging.info(
        f"HLG after k2.connect: #states: {HLG.shape[0]}, #arcs: {HLG.num_arcs}"
    )

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)

    return HLG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    filename = lang_dir / "HLG_fst.pt"

    if filename.is_file():
        logging.info(f"{filename} already exists - skipping")
        return

    HLG = compile_HLG(lang_dir, args.lm)
    logging.info(f"Saving HLG to {filename}")
    torch.save(HLG.as_dict(), filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
