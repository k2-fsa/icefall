#!/usr/bin/env python3

# Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

"""
This script takes as input lang_dir containing lexicon_disambig.txt,
tokens.txt, and words.txt and generates the following files:

    - H.fst
    - HL.fst
    - HLG.fst

Note that saved files are in OpenFst binary format.

Usage:

./local/prepare_lang_fst.py \
  --lang-dir ./data/lang_phone \
  --has-silence 1

Or

./local/prepare_lang_fst.py \
  --lang-dir ./data/lang_bpe_500
"""

import argparse
import logging
from pathlib import Path

import kaldifst

from icefall.ctc import (
    Lexicon,
    add_disambig_self_loops,
    add_one,
    build_standard_ctc_topo,
    make_lexicon_fst_no_silence,
    make_lexicon_fst_with_silence,
)
from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        """,
    )

    parser.add_argument(
        "--has-silence",
        type=str2bool,
        default=False,
        help="True if the lexicon has silence.",
    )

    parser.add_argument(
        "--ngram-G",
        type=str,
        help="""If not empty, it is the filename of G used to build HLG.
        For instance, --ngram-G=./data/lm/G_3_fst.txt
        """,
    )

    return parser.parse_args()


def build_HL(
    H: kaldifst.StdVectorFst,
    L: kaldifst.StdVectorFst,
    has_silence: bool,
    lexicon: Lexicon,
) -> kaldifst.StdVectorFst:
    if has_silence:
        # We also need to change the input labels of L
        add_one(L, treat_ilabel_zero_specially=True, update_olabel=False)
    else:
        add_one(L, treat_ilabel_zero_specially=False, update_olabel=False)

    # Invoke add_disambig_self_loops() so that it eats the disambig symbols
    # from L after composition
    add_disambig_self_loops(
        H,
        start=lexicon.token2id["#0"] + 1,
        end=lexicon.max_disambig_id + 1,
    )

    kaldifst.arcsort(H, sort_type="olabel")
    kaldifst.arcsort(L, sort_type="ilabel")

    HL = kaldifst.compose(H, L)
    kaldifst.determinize_star(HL)

    disambig0 = lexicon.token2id["#0"] + 1
    max_disambig = lexicon.max_disambig_id + 1
    for state in kaldifst.StateIterator(HL):
        for arc in kaldifst.ArcIterator(HL, state):
            # If treat_ilabel_zero_specially is False, we always change it
            # Otherwise, we only change non-zero input labels
            if disambig0 <= arc.ilabel <= max_disambig:
                arc.ilabel = 0

    # Note: We are not composing L with G, so there is no need to add
    # self-loops to L to handle #0

    return HL


def build_HLG(
    H: kaldifst.StdVectorFst,
    L: kaldifst.StdVectorFst,
    G: kaldifst.StdVectorFst,
    has_silence: bool,
    lexicon: Lexicon,
) -> kaldifst.StdVectorFst:
    if has_silence:
        # We also need to change the input labels of L
        add_one(L, treat_ilabel_zero_specially=True, update_olabel=False)
    else:
        add_one(L, treat_ilabel_zero_specially=False, update_olabel=False)

    # add-self-loops
    token_disambig0 = lexicon.token2id["#0"] + 1
    word_disambig0 = lexicon.word2id["#0"]

    kaldifst.add_self_loops(L, isyms=[token_disambig0], osyms=[word_disambig0])

    kaldifst.arcsort(L, sort_type="olabel")
    kaldifst.arcsort(G, sort_type="ilabel")
    LG = kaldifst.compose(L, G)
    kaldifst.determinize_star(LG)
    kaldifst.minimize_encoded(LG)

    kaldifst.arcsort(LG, sort_type="ilabel")

    # Invoke add_disambig_self_loops() so that it eats the disambig symbols
    # from L after composition
    add_disambig_self_loops(
        H,
        start=lexicon.token2id["#0"] + 1,
        end=lexicon.max_disambig_id + 1,
    )

    kaldifst.arcsort(H, sort_type="olabel")

    HLG = kaldifst.compose(H, LG)
    kaldifst.determinize_star(HLG)

    disambig0 = lexicon.token2id["#0"] + 1
    max_disambig = lexicon.max_disambig_id + 1
    for state in kaldifst.StateIterator(HLG):
        for arc in kaldifst.ArcIterator(HLG, state):
            # If treat_ilabel_zero_specially is False, we always change it
            # Otherwise, we only change non-zero input labels
            if disambig0 <= arc.ilabel <= max_disambig:
                arc.ilabel = 0
    return HLG


def copy_fst(fst):
    # Please don't use fst.copy()
    return kaldifst.StdVectorFst(fst)


def main():
    args = get_args()
    lang_dir = args.lang_dir

    lexicon = Lexicon(lang_dir)

    logging.info("Building standard CTC topology")
    max_token_id = max(lexicon.tokens)
    H = build_standard_ctc_topo(max_token_id=max_token_id)

    # We need to add one to all tokens since we want to use ID 0
    # for epsilon
    add_one(H, treat_ilabel_zero_specially=False, update_olabel=True)
    H.write(f"{lang_dir}/H.fst")

    logging.info("Building L")
    # Now for HL

    if args.has_silence:
        L = make_lexicon_fst_with_silence(lexicon, attach_symbol_table=False)
    else:
        L = make_lexicon_fst_no_silence(lexicon, attach_symbol_table=False)

    logging.info("Building HL")
    HL = build_HL(
        H=copy_fst(H),
        L=copy_fst(L),
        has_silence=args.has_silence,
        lexicon=lexicon,
    )
    HL.write(f"{lang_dir}/HL.fst")

    if not args.ngram_G:
        logging.info("Skip building HLG")
        return

    logging.info("Building HLG")
    with open(args.ngram_G) as f:
        G = kaldifst.compile(
            s=f.read(),
            acceptor=False,
        )

    HLG = build_HLG(H=H, L=L, G=G, has_silence=args.has_silence, lexicon=lexicon)
    HLG.write(f"{lang_dir}/HLG.fst")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
