#!/usr/bin/env python3

# Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as input data/lang_phone containing lexicon_disambig.txt,
tokens.txt, and words.txt and generates the following files:

    - H.fst
    - HL.fst

TODO(fangjun): Generate HLG.fst

Note that saved files are in OpenFst binary format.
"""

from pathlib import Path

import kaldifst

from icefall.ctc import (
    Lexicon,
    add_disambig_self_loops,
    add_one,
    build_standard_ctc_topo,
    make_lexicon_fst_with_silence,
)


def main():
    lang_dir = Path("data/lang_phone")
    lexicon = Lexicon(lang_dir)

    max_token_id = max(lexicon.tokens)
    H = build_standard_ctc_topo(max_token_id=max_token_id)

    # We need to add one to all tokens since we want to use ID 0
    # for epsilon
    add_one(H, treat_ilabel_zero_specially=False, update_olabel=True)
    H.write(f"{lang_dir}/H.fst")

    # Now for HL
    L = make_lexicon_fst_with_silence(lexicon, attach_symbol_table=False)

    # We also need to change the input labels of L
    add_one(L, treat_ilabel_zero_specially=True, update_olabel=False)

    # Invoke add_disambig_self_loops() so that it eats the disambig symbols
    # from L after composition
    add_disambig_self_loops(
        H,
        start=lexicon.token2id["#0"] + 1,
        end=lexicon.max_disambig_id,
    )

    kaldifst.arcsort(H, sort_type="olabel")
    kaldifst.arcsort(L, sort_type="ilabel")
    HL = kaldifst.compose(H, L)

    # Note: We are not composing L with G, so there is no need to add
    # self-loops to L to handle #0

    HL.write(f"{lang_dir}/HL.fst")


if __name__ == "__main__":
    main()
