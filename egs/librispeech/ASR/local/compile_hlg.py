#!/usr/bin/env python3

"""
This script compiles HLG from

    - H, the ctc topology, built from tokens contained in lexicon.txt
    - L, the lexicon, built from L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated HLG is saved in data/lm/HLG.pt (phone based)
or data/lm/HLG_bpe.pt (BPE based)
"""
import logging
from pathlib import Path

import k2
import torch

from icefall.lexicon import Lexicon


def compile_HLG(lang_dir: str) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang or data/lang/bpe.

    Return:
      An FSA representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path("data/lm/G_3_gram.pt").is_file():
        logging.info("Loading pre-compiled G_3_gram")
        d = torch.load("data/lm/G_3_gram.pt")
        G = k2.Fsa.from_dict(d)
    else:
        logging.info("Loading G_3_gram.fst.txt")
        with open("data/lm/G_3_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), "G_3_gram.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    LG.labels[LG.labels >= first_token_disambig_id] = 0

    assert isinstance(LG.aux_labels, k2.RaggedInt)
    LG.aux_labels.values()[LG.aux_labels.values() >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing H and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"HLG.shape: {HLG.shape}")

    return HLG


def phone_based_HLG():
    if Path("data/lm/HLG.pt").is_file():
        return

    logging.info("Compiling phone based HLG")
    HLG = compile_HLG("data/lang")

    logging.info("Saving HLG.pt to data/lm")
    torch.save(HLG.as_dict(), "data/lm/HLG.pt")


def bpe_based_HLG():
    if Path("data/lm/HLG_bpe.pt").is_file():
        return

    logging.info("Compiling BPE based HLG")
    HLG = compile_HLG("data/lang/bpe")
    logging.info("Saving HLG_bpe.pt to data/lm")
    torch.save(HLG.as_dict(), "data/lm/HLG_bpe.pt")


def main():
    phone_based_HLG()
    bpe_based_HLG()


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
