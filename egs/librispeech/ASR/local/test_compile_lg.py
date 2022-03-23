#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./local/test_compile_lg.py
"""

import os

from pathlib import Path

import k2
import torch

lang_dir = Path("./data/lang_bpe_500")
corpus = "test_compile_lg_corpus.txt"
arpa = "test_compile_lg_3_gram.arpa"
G_fst_txt = "test_compile_lg_3_gram.fst.txt"


def generate_corpus():
    s = """HELLO WORLD
HELLOA WORLDER
HELLOA WORLDER HELLO
HELLOA WORLDER"""
    with open(corpus, "w") as f:
        f.write(s)


def generate_arpa():
    cmd = f"""
      ./shared/make_kn_lm.py \
        -ngram-order 3 \
        -text {corpus} \
        -lm {arpa}
    """
    os.system(cmd)


def generate_G():
    cmd = f"""
      python3 -m kaldilm \
        --read-symbol-table="{lang_dir}/words.txt" \
        --disambig-symbol='#0' \
        {arpa} > {G_fst_txt}
    """
    os.system(cmd)


def main():
    generate_corpus()
    generate_arpa()
    generate_G()
    with open(G_fst_txt) as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        del G.aux_labels
    G.labels_sym = k2.SymbolTable.from_file(f"{lang_dir}/words.txt")
    G.draw("G.pdf", title="G")

    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))
    L.labels_sym = k2.SymbolTable.from_file(f"{lang_dir}/tokens.txt")
    L.aux_labels_sym = k2.SymbolTable.from_file(f"{lang_dir}/words.txt")

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    LG = k2.compose(L, G)
    del LG.aux_labels

    LG = k2.determinize(LG)
    LG = k2.connect(LG)
    LG = k2.arc_sort(LG)
    print(LG.properties_str)
    LG.draw("LG.pdf", title="LG")
    # You can have a look at G.pdf and LG.pdf to get a feel
    # what they look like


if __name__ == "__main__":
    main()
