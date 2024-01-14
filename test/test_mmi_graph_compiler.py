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

import copy
import os
import shutil
import sys
from pathlib import Path

import k2
import sentencepiece as spm

from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler

TMP_DIR = "/tmp/icefall-test-mmi-graph-compiler"
USING_PYTEST = "pytest" in sys.modules
ICEFALL_DIR = Path(__file__).resolve().parent.parent


def generate_test_data():
    Path(TMP_DIR).mkdir(exist_ok=True)
    sentences = """
cat tac cat cat
at at cat at cat cat
tac at ta at at
at cat ct ct ta ct ct cat tac
cat cat cat cat
at at at at at at at
    """

    transcript = Path(TMP_DIR) / "transcript_words.txt"
    with open(transcript, "w") as f:
        for line in sentences.strip().split("\n"):
            f.write(f"{line}\n")

    words = """
<eps> 0
<UNK> 1
at 2
cat 3
ct 4
ta 5
tac 6
#0 7
<s> 8
</s> 9
"""
    word_txt = Path(TMP_DIR) / "words.txt"
    with open(word_txt, "w") as f:
        for line in words.strip().split("\n"):
            f.write(f"{line}\n")

    vocab_size = 8

    os.system(
        f"""
cd {ICEFALL_DIR}/egs/librispeech/ASR

./local/train_bpe_model.py \
  --lang-dir {TMP_DIR} \
  --vocab-size {vocab_size} \
  --transcript {transcript}

./local/prepare_lang_bpe.py --lang-dir {TMP_DIR} --debug 0

./local/convert_transcript_words_to_tokens.py \
--lexicon {TMP_DIR}/lexicon.txt \
--transcript {transcript} \
--oov "<UNK>" \
> {TMP_DIR}/transcript_tokens.txt

./shared/make_kn_lm.py \
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
    # Caution:
    # You have to uncomment
    #  del transcript_fsa.aux_labels
    # in mmi_graph_compiler.py
    # to see the correct aux_labels in *.svg
    graph_compiler = MmiTrainingGraphCompiler(
        lang_dir=TMP_DIR, uniq_filename="lexicon.txt"
    )
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

    texts = ["cat at ct", "at ta", "cat tac"]
    transcript_fsa = graph_compiler.build_transcript_fsa(texts)
    transcript_fsa[0].draw(f"{TMP_DIR}/cat_at_ct.svg", title="cat_at_ct")
    transcript_fsa[1].draw(f"{TMP_DIR}/at_ta.svg", title="at_ta")
    transcript_fsa[2].draw(f"{TMP_DIR}/cat_tac.svg", title="cat_tac")

    num_graphs, den_graphs = graph_compiler.compile(texts, replicate_den=True)
    num_graphs[0].draw(f"{TMP_DIR}/num_cat_at_ct.svg", title="num_cat_at_ct")
    num_graphs[1].draw(f"{TMP_DIR}/num_at_ta.svg", title="num_at_ta")
    num_graphs[2].draw(f"{TMP_DIR}/num_cat_tac.svg", title="num_cat_tac")

    den_graphs[0].draw(f"{TMP_DIR}/den_cat_at_ct.svg", title="den_cat_at_ct")
    den_graphs[2].draw(f"{TMP_DIR}/den_cat_tac.svg", title="den_cat_tac")

    sp = spm.SentencePieceProcessor()
    sp.load(f"{TMP_DIR}/bpe.model")

    texts = ["cat at cat", "at tac"]
    token_ids = graph_compiler.texts_to_ids(texts)
    expected_token_ids = sp.encode(texts)
    assert token_ids == expected_token_ids


def test_main():
    generate_test_data()

    mmi_graph_compiler_test()

    if USING_PYTEST:
        delete_test_data()


def main():
    test_main()


if __name__ == "__main__" and not USING_PYTEST:
    main()
