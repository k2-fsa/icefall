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

    (1) cd icefall; pytest test/test_lexicon.py
    (2) cd icefall; ./test/test_lexicon.py
"""


import os
import shutil
import sys
from pathlib import Path

import k2

from icefall.lexicon import UniqLexicon

TMP_DIR = "/tmp/icefall-test-lexicon"
USING_PYTEST = "pytest" in sys.modules
ICEFALL_DIR = Path(__file__).resolve().parent.parent


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

    os.system(
        f"""
cd {ICEFALL_DIR}/egs/librispeech/ASR

./local/generate_unique_lexicon.py --lang-dir {TMP_DIR}
./local/prepare_lang.py --lang-dir {TMP_DIR}
"""
    )


def delete_test_data():
    shutil.rmtree(TMP_DIR)


def uniq_lexicon_test():
    lexicon = UniqLexicon(lang_dir=TMP_DIR, uniq_filename="uniq_lexicon.txt")

    texts = ["cat cat", "at ac", "ca at cat"]
    token_ids = lexicon.texts_to_token_ids(texts)
    #
    #                c  a  t  c  a  t    a  t  a  3   SPN a  t  c  a  t
    expected_ids = [[3, 2, 4, 3, 2, 4], [2, 4, 2, 3], [1, 2, 4, 3, 2, 4]]
    expected_ids = k2.RaggedTensor(expected_ids)

    assert token_ids == expected_ids


def test_main():
    generate_test_data()

    uniq_lexicon_test()

    if USING_PYTEST:
        delete_test_data()


def main():
    test_main()


if __name__ == "__main__" and not USING_PYTEST:
    main()
