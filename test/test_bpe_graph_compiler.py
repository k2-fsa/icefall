#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.lexicon import BpeLexicon
from pathlib import Path


def test():
    lang_dir = Path("data/lang/bpe")
    if not lang_dir.is_dir():
        return
    # TODO: generate data for testing

    compiler = BpeCtcTrainingGraphCompiler(lang_dir)
    ids = compiler.texts_to_ids(["HELLO", "WORLD ZZZ"])
    fsa = compiler.compile(ids)

    lexicon = BpeLexicon(lang_dir)
    ids0 = lexicon.words_to_piece_ids(["HELLO"])
    assert ids[0] == ids0.values().tolist()

    ids1 = lexicon.words_to_piece_ids(["WORLD", "ZZZ"])
    assert ids[1] == ids1.values().tolist()
