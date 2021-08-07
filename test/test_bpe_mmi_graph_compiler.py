#!/usr/bin/env python3

import copy
import logging
from pathlib import Path

import k2
import torch

from icefall.bpe_mmi_graph_compiler import BpeMmiTrainingGraphCompiler


def test_bpe_mmi_graph_compiler():
    lang_dir = Path("data/lang_bpe")
    if lang_dir.is_dir() is False:
        return
    device = torch.device("cpu")
    compiler = BpeMmiTrainingGraphCompiler(lang_dir, device=device)

    texts = ["HELLO WORLD", "MMI TRAINING"]

    num_graphs, den_graphs = compiler.compile(texts)
    num_graphs.labels_sym = compiler.lexicon.token_table
    num_graphs.aux_labels_sym = copy.deepcopy(compiler.lexicon.token_table)
    num_graphs.aux_labels_sym._id2sym[0] = "<eps>"
    num_graphs[0].draw("num_graphs_0.svg", title="HELLO WORLD")
    num_graphs[1].draw("num_graphs_1.svg", title="HELLO WORLD")
    print(den_graphs.shape)
    print(den_graphs[0].shape)
    print(den_graphs[0].num_arcs)
