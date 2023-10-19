#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corp. (authors: Fangjun Kuang)
# See ../../LICENSE for clarification regarding multiple authors
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import k2
from icefall.decode import Nbest

def test_nbest_from_lattice():
    s = """
        0 1 1 10 0.1
        0 1 5 10 0.11
        0 1 2 20 0.2
        1 2 3 30 0.3
        1 2 4 40 0.4
        2 3 -1 -1 0.5
        3
    """
    lattice = k2.Fsa.from_str(s, acceptor=False)
    lattice = k2.Fsa.from_fsas([lattice, lattice])

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=10,
        use_double_scores=True,
        nbest_scale=0.5
    )

    assert nbest.fsa.shape[0] == 4 * 2
    assert nbest.shape.row_splits(1).tolist() == [0, 4, 8]

    nbest2 = nbest.intersect(lattice)
    tot_scores = nbest2.tot_scores()
    argmax = tot_scores.argmax()
    best_path = k2.index_fsa(nbest2.fsa, argmax)
    print(best_path[0])
