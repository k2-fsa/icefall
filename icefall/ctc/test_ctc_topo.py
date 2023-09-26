#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

from pathlib import Path

import graphviz
import kaldifst
import sentencepiece as spm
from prepare_lang import (
    Lexicon,
    make_lexicon_fst_no_silence,
    make_lexicon_fst_with_silence,
)
from topo import add_disambig_self_loops, add_one, build_standard_ctc_topo


def test_yesno():
    lang_dir = "/Users/fangjun/open-source/icefall/egs/yesno/ASR/data/lang_phone"
    if not Path(lang_dir).is_dir():
        print(f"{lang_dir} does not exist! Skip testing")
        return
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)

    H = build_standard_ctc_topo(max_token_id=max_token_id)

    isym = kaldifst.SymbolTable()
    isym.add_symbol(symbol="<blk>", key=0)
    for i in range(1, max_token_id + 1):
        isym.add_symbol(symbol=lexicon.id2token[i], key=i)

    osym = kaldifst.SymbolTable()
    osym.add_symbol(symbol="<eps>", key=0)
    for i in range(1, max_token_id + 1):
        osym.add_symbol(symbol=lexicon.id2token[i], key=i)

    H.input_symbols = isym
    H.output_symbols = osym

    fst_dot = kaldifst.draw(H, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="standard_ctc_topo_yesno.pdf")
    # See the link below to visualize the above PDF
    # https://t.ly/7uXZ9

    # Now test HL

    # We need to add one to all tokens since we want to use ID 0
    # for epsilon
    add_one(H, treat_ilabel_zero_specially=False, update_olabel=True)

    add_disambig_self_loops(
        H,
        start=lexicon.token2id["#0"] + 1,
        end=lexicon.max_disambig_id,
    )

    fst_dot = kaldifst.draw(H, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="standard_ctc_topo_disambig_yesno.pdf")

    L = make_lexicon_fst_with_silence(lexicon)

    # We also need to change the input labels of L
    add_one(L, treat_ilabel_zero_specially=True, update_olabel=False)

    H.output_symbols = None

    kaldifst.arcsort(H, sort_type="olabel")
    kaldifst.arcsort(L, sort_type="ilabel")
    HL = kaldifst.compose(H, L)

    lexicon.id2token[0] = "<blk>"
    lexicon.token2id["<blk>"] = 0

    isym = kaldifst.SymbolTable()
    isym.add_symbol(symbol="<eps>", key=0)
    for i in range(0, lexicon.max_disambig_id + 1):
        isym.add_symbol(symbol=lexicon.id2token[i], key=i + 1)

    osym = kaldifst.SymbolTable()
    for i, word in lexicon.id2word.items():
        osym.add_symbol(symbol=word, key=i)

    HL.input_symbols = isym
    HL.output_symbols = osym

    fst_dot = kaldifst.draw(HL, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="HL_yesno.pdf")


def test_librispeech():
    lang_dir = (
        "/star-fj/fangjun/open-source/icefall-2/egs/librispeech/ASR/data/lang_bpe_500"
    )

    if not Path(lang_dir).is_dir():
        print(f"{lang_dir} does not exist! Skip testing")
        return

    lexicon = Lexicon(lang_dir)
    HL = kaldifst.StdVectorFst.read(lang_dir + "/HL.fst")

    sp = spm.SentencePieceProcessor()
    sp.load(lang_dir + "/bpe.model")

    i = lexicon.word2id["HELLOA"]
    k = lexicon.word2id["WORLD"]
    print(i, k)
    s = f"""
        0 1 {i} {i}
        1 2 {k} {k}
        2
    """
    fst = kaldifst.compile(
        s=s,
        acceptor=False,
    )

    L = make_lexicon_fst_no_silence(lexicon, attach_symbol_table=False)
    kaldifst.arcsort(L, sort_type="olabel")
    with open("L.fst.txt", "w") as f:
        print(L, file=f)

    fst = kaldifst.compose(L, fst)
    print(fst)
    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="a.pdf")
    print(sp.encode(["HELLOA", "WORLD"]))


def main():
    test_yesno()
    test_librispeech()


if __name__ == "__main__":
    main()
