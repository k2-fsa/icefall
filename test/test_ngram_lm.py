#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../LICENSE for clarification regarding multiple authors
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

import graphviz

from icefall import is_module_available

if not is_module_available("kaldifst"):
    raise ValueError("Please 'pip install kaldifst' first.")

import kaldifst

from icefall import NgramLm, NgramLmStateCost


def generate_fst(filename: str):
    s = """
3	5	1	1	3.00464
3	0	3	0	5.75646
0	1	1	1	12.0533
0	2	2	2	7.95954
0	9.97787
1	4	2	2	3.35436
1	0	3	0	7.59853
2	0	3	0
4	2	3	0	7.43735
4	0.551239
5	4	2	2	0.804938
5	1	3	0	9.67086
"""
    fst = kaldifst.compile(s=s, acceptor=False)
    fst.write(filename)
    fst_dot = kaldifst.draw(fst, acceptor=False, portrait=True)
    source = graphviz.Source(fst_dot)
    source.render(outfile=f"{filename}.svg")


def main():
    filename = "test.fst"
    generate_fst(filename)
    ngram_lm = NgramLm(filename, backoff_id=3, is_binary=True)
    for label in [1, 2, 3, 4, 5]:
        print("---label---", label)
        p = ngram_lm.get_next_state_and_cost(state=5, label=label)
        print(p)
        print("---")

    state_cost = NgramLmStateCost(ngram_lm)
    s0 = state_cost.forward_one_step(1)
    print(s0.state_cost)

    s1 = s0.forward_one_step(2)
    print(s1.state_cost)

    s2 = s1.forward_one_step(2)
    print(s2.state_cost)


if __name__ == "__main__":
    main()
