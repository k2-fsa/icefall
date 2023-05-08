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
from typing import List

import sentencepiece as spm

from icefall.lexicon import UniqLexicon

TMP_DIR = "/tmp/icefall-test-lexicon"
USING_PYTEST = "pytest" in sys.modules
ICEFALL_DIR = Path(__file__).resolve().parent.parent


def generate_test_data():
    Path(TMP_DIR).mkdir(exist_ok=True)
    sentences = """
cat tac cat cat
at
tac at ta at at
at cat ct ct ta
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

./local/prepare_lang_bpe.py --lang-dir {TMP_DIR} --debug 1
"""
    )


def delete_test_data():
    shutil.rmtree(TMP_DIR)


def uniq_lexicon_test():
    lexicon = UniqLexicon(lang_dir=TMP_DIR, uniq_filename="lexicon.txt")

    # case 1: No OOV
    texts = ["cat cat", "at ct", "at tac cat"]
    token_ids = lexicon.texts_to_token_ids(texts)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{TMP_DIR}/bpe.model")

    expected_token_ids: List[List[int]] = sp.encode(texts, out_type=int)
    assert token_ids.tolist() == expected_token_ids

    # case 2: With OOV
    texts = ["ca"]
    token_ids = lexicon.texts_to_token_ids(texts)
    expected_token_ids = sp.encode(texts, out_type=int)
    assert token_ids.tolist() != expected_token_ids
    # Note: sentencepiece breaks "ca" into "_ c a"
    # But there is no word "ca" in the lexicon, so our
    # implementation returns the id of "<UNK>"
    print(token_ids, expected_token_ids)
    assert token_ids.tolist() == [[sp.piece_to_id("▁"), sp.unk_id()]]

    # case 3: With OOV
    texts = ["foo"]
    token_ids = lexicon.texts_to_token_ids(texts)
    expected_token_ids = sp.encode(texts, out_type=int)
    print(token_ids)
    print(expected_token_ids)

    # test ragged lexicon
    ragged_lexicon = lexicon.ragged_lexicon.tolist()
    word_disambig_id = lexicon.word_table["#0"]
    for i in range(2, word_disambig_id):
        piece_id = ragged_lexicon[i]
        word = lexicon.word_table[i]
        assert word == sp.decode(piece_id)
        assert piece_id == sp.encode(word)


def test_main():
    generate_test_data()

    uniq_lexicon_test()

    if USING_PYTEST:
        delete_test_data()


def main():
    test_main()


if __name__ == "__main__" and not USING_PYTEST:
    main()
