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


from pathlib import Path

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.lexicon import UniqLexicon

ICEFALL_DIR = Path(__file__).resolve().parent.parent


def test():
    lang_dir = ICEFALL_DIR / "egs/librispeech/ASR/data/lang_bpe"
    if not lang_dir.is_dir():
        return

    compiler = BpeCtcTrainingGraphCompiler(lang_dir)
    ids = compiler.texts_to_ids(["HELLO", "WORLD ZZZ"])
    compiler.compile(ids)

    lexicon = UniqLexicon(lang_dir, uniq_filename="lexicon.txt")
    ids0 = lexicon.words_to_piece_ids(["HELLO"])
    assert ids[0] == ids0.values().tolist()

    ids1 = lexicon.words_to_piece_ids(["WORLD", "ZZZ"])
    assert ids[1] == ids1.values().tolist()
