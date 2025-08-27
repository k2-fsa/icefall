#!/usr/bin/env python3
# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
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
You can run this file in one of the two ways:

    (1) cd icefall; pytest test/test_decode.py
    (2) cd icefall; ./test/test_decode.py
"""

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
        nbest_scale=0.5,
    )
    # each lattice has only 4 distinct paths that have different word sequences:
    # 10->30
    # 10->40
    # 20->30
    # 20->40
    #
    # So there should be only 4 paths for each lattice in the Nbest object
    assert nbest.fsa.shape[0] == 4 * 2
    assert nbest.shape.row_splits(1).tolist() == [0, 4, 8]

    nbest2 = nbest.intersect(lattice)
    tot_scores = nbest2.tot_scores()
    argmax = tot_scores.argmax()
    best_path = k2.index_fsa(nbest2.fsa, argmax)
    print(best_path[0])
