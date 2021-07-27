#!/usr/bin/env python3

"""
This script compiles HLG from

    - H, the ctc topology, built from phones contained in lexicon.txt
    - L, the lexicon, built from L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated HLG is saved in data/lm/HLG.pt (phone based)
or data/lm/HLG_bpe.pt (BPE based)
"""
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
    print(f"building ctc_top. max_token_id: {max_token_id}")
    H = k2.ctc_topo(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    if Path("data/lm/G_3_gram.pt").is_file():
        print("Loading pre-compiled G_3_gram")
        d = torch.load("data/lm/G_3_gram.pt")
        G = k2.Fsa.from_dict(d)
    else:
        print("Loading G_3_gram.fst.txt")
        with open("data/lm/G_3_gram.fst.txt") as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
            torch.save(G.as_dict(), "G_3_gram.pt")

    first_token_disambig_id = lexicon.phones["#0"]
    first_word_disambig_id = lexicon.words["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    print("Intersecting L and G")
    LG = k2.compose(L, G)
    print(f"LG shape: {LG.shape}")

    print("Connecting LG")
    LG = k2.connect(LG)
    print(f"LG shape after k2.connect: {LG.shape}")

    print(type(LG.aux_labels))
    print("Determinizing LG")

    LG = k2.determinize(LG)
    print(type(LG.aux_labels))

    print("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    print("Removing disambiguation symbols on LG")

    LG.labels[LG.labels >= first_token_disambig_id] = 0

    assert isinstance(LG.aux_labels, k2.RaggedInt)
    LG.aux_labels.values()[LG.aux_labels.values() >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    print(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    print("Arc sorting LG")
    LG = k2.arc_sort(LG)

    print("Composing H and LG")
    HLG = k2.compose(H, LG, inner_labels="phones")

    print("Connecting LG")
    HLG = k2.connect(HLG)

    print("Arc sorting LG")
    HLG = k2.arc_sort(HLG)

    return HLG


def phone_based_HLG():
    if Path("data/lm/HLG.pt").is_file():
        return

    print("Compiling phone based HLG")
    HLG = compile_HLG("data/lang")

    print("Saving HLG.pt to data/lm")
    torch.save(HLG.as_dict(), "data/lm/HLG.pt")


def bpe_based_HLG():
    if Path("data/lm/HLG_bpe.pt").is_file():
        return

    print("Compiling BPE based HLG")
    HLG = compile_HLG("data/lang/bpe")
    print("Saving HLG_bpe.pt to data/lm")
    torch.save(HLG.as_dict(), "data/lm/HLG_bpe.pt")


def main():
    phone_based_HLG()
    bpe_based_HLG()


if __name__ == "__main__":
    main()
