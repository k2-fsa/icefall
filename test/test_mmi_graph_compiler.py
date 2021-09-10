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

    (1) cd icefall; pytest test/test_mmi_graph_compiler.py
    (2) cd icefall; ./test/test_mmi_graph_compiler.py
"""

import os
import shutil
import sys
import copy
from pathlib import Path

import k2

from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler

TMP_DIR = "/tmp/icefall-test-mmi-graph-compiler"
USING_PYTEST = "pytest" in sys.modules
ICEFALL_DIR = Path(__file__).resolve().parent.parent
print(ICEFALL_DIR)


def generate_test_data():
    #  if Path(TMP_DIR).exists():
    #      return
    Path(TMP_DIR).mkdir(exist_ok=True)
    lexicon = """
<UNK> SPN
cat c a t
at a t
at a a t
ac a c
ac a c c
"""
    lexicon_filename = Path(TMP_DIR) / "lexicon.txt"
    with open(lexicon_filename, "w") as f:
        for line in lexicon.strip().split("\n"):
            f.write(f"{line}\n")
    transcript_words = """
cat at ta
at at cat ta
"""
    transcript_words_filename = Path(TMP_DIR) / "transcript_words.txt"
    with open(transcript_words_filename, "w") as f:
        for line in transcript_words.strip().split("\n"):
            f.write(f"{line}\n")
    os.system(
        f"""
cd {ICEFALL_DIR}/egs/librispeech/ASR

./local/generate_unique_lexicon.py --lang-dir {TMP_DIR}
./local/prepare_lang.py --lang-dir {TMP_DIR}

./local/convert_transcript_words_to_tokens.py \
  --lexicon {TMP_DIR}/uniq_lexicon.txt \
  --transcript {TMP_DIR}/transcript_words.txt \
  --oov "<UNK>" \
  > {TMP_DIR}/transcript_tokens.txt

shared/make_kn_lm.py \
  -ngram-order 2 \
  -text {TMP_DIR}/transcript_tokens.txt \
  -lm {TMP_DIR}/P.arpa

python3 -m kaldilm \
  --read-symbol-table="{TMP_DIR}/tokens.txt" \
  --disambig-symbol='#0' \
  --max-order=2 \
  {TMP_DIR}/P.arpa > {TMP_DIR}/P.fst.txt
"""
    )


def delete_test_data():
    shutil.rmtree(TMP_DIR)


def mmi_graph_compiler_test():
    graph_compiler = MmiTrainingGraphCompiler(lang_dir=TMP_DIR)
    print(graph_compiler.device)
    L_inv = graph_compiler.L_inv
    L = k2.invert(L_inv)

    L.labels_sym = graph_compiler.lexicon.token_table
    L.aux_labels_sym = graph_compiler.lexicon.word_table
    L.draw(f"{TMP_DIR}/L.svg", title="L")

    L_inv.labels_sym = graph_compiler.lexicon.word_table
    L_inv.aux_labels_sym = graph_compiler.lexicon.token_table
    L_inv.draw(f"{TMP_DIR}/L_inv.svg", title="L")

    ctc_topo_P = graph_compiler.ctc_topo_P
    ctc_topo_P.labels_sym = copy.deepcopy(graph_compiler.lexicon.token_table)
    ctc_topo_P.labels_sym._id2sym[0] = "<blk>"
    ctc_topo_P.labels_sym._sym2id["<blk>"] = 0
    ctc_topo_P.aux_labels_sym = graph_compiler.lexicon.token_table
    ctc_topo_P.draw(f"{TMP_DIR}/ctc_topo_P.svg", title="ctc_topo_P")

    print(ctc_topo_P.num_arcs)
    print(k2.connect(ctc_topo_P).num_arcs)

    with open(str(TMP_DIR) + "/P.fst.txt") as f:
        # P is not an acceptor because there is
        # a back-off state, whose incoming arcs
        # have label #0 and aux_label 0 (i.e., <eps>).
        P = k2.Fsa.from_openfst(f.read(), acceptor=False)
    P.labels_sym = graph_compiler.lexicon.token_table
    P.aux_labels_sym = graph_compiler.lexicon.token_table
    P.draw(f"{TMP_DIR}/P.svg", title="P")

    ctc_topo = k2.ctc_topo(max(graph_compiler.lexicon.tokens), False)
    ctc_topo.labels_sym = ctc_topo_P.labels_sym
    ctc_topo.aux_labels_sym = graph_compiler.lexicon.token_table
    ctc_topo.draw(f"{TMP_DIR}/ctc_topo.svg", title="ctc_topo")
    print("p num arcs", P.num_arcs)
    print("ctc_topo num arcs", ctc_topo.num_arcs)
    print("ctc_topo_P num arcs", ctc_topo_P.num_arcs)

    texts = ["cat at ac at", "at ac cat zoo", "cat zoo"]
    transcript_fsa = graph_compiler.build_transcript_fsa(texts)
    transcript_fsa[0].draw(f"{TMP_DIR}/cat_at_ac_at.svg", title="cat_at_ac_at")
    transcript_fsa[1].draw(
        f"{TMP_DIR}/at_ac_cat_zoo.svg", title="at_ac_cat_zoo"
    )
    transcript_fsa[2].draw(f"{TMP_DIR}/cat_zoo.svg", title="cat_zoo")

    num_graphs, den_graphs = graph_compiler.compile(texts, replicate_den=True)
    num_graphs[0].draw(
        f"{TMP_DIR}/num_cat_at_ac_at.svg", title="num_cat_at_ac_at"
    )
    num_graphs[1].draw(
        f"{TMP_DIR}/num_at_ac_cat_zoo.svg", title="num_at_ac_cat_zoo"
    )
    num_graphs[2].draw(f"{TMP_DIR}/num_cat_zoo.svg", title="num_cat_zoo")

    den_graphs[0].draw(
        f"{TMP_DIR}/den_cat_at_ac_at.svg", title="den_cat_at_ac_at"
    )
    den_graphs[2].draw(f"{TMP_DIR}/den_cat_zoo.svg", title="den_cat_zoo")


def test_main():
    generate_test_data()

    mmi_graph_compiler_test()

    if USING_PYTEST:
        delete_test_data()


def main():
    test_main()


if __name__ == "__main__" and not USING_PYTEST:
    main()
