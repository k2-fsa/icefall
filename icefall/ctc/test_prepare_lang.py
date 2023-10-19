#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

from pathlib import Path

import graphviz
import kaldifst
from prepare_lang import Lexicon, make_lexicon_fst_with_silence


def test_yesno():
    lang_dir = "/Users/fangjun/open-source/icefall/egs/yesno/ASR/data/lang_phone"
    if not Path(lang_dir).is_dir():
        print(f"{lang_dir} does not exist! Skip testing")
        return

    lexicon = Lexicon(lang_dir)

    L = make_lexicon_fst_with_silence(lexicon)

    isym = kaldifst.SymbolTable()
    for i, token in lexicon.id2token.items():
        isym.add_symbol(symbol=token, key=i)

    osym = kaldifst.SymbolTable()
    for i, word in lexicon.id2word.items():
        osym.add_symbol(symbol=word, key=i)

    L.input_symbols = isym
    L.output_symbols = osym
    fst_dot = kaldifst.draw(L, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile="L_yesno.pdf")
    # See the link below to visualize the above PDF
    # https://t.ly/jMfXW


def main():
    test_yesno()


if __name__ == "__main__":
    main()
