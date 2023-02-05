#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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


# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

import os
import tempfile

import k2
from prepare_lang import (
    add_disambig_symbols,
    generate_id_map,
    get_phones,
    get_words,
    lexicon_to_fst,
    read_lexicon,
    write_lexicon,
    write_mapping,
)


def generate_lexicon_file() -> str:
    fd, filename = tempfile.mkstemp()
    os.close(fd)
    s = """
    !SIL SIL
    <SPOKEN_NOISE> SPN
    <UNK> SPN
    f f
    a a
    foo f o o
    bar b a r
    bark b a r k
    food f o o d
    food2 f o o d
    fo  f o
    """.strip()
    with open(filename, "w") as f:
        f.write(s)
    return filename


def test_read_lexicon(filename: str):
    lexicon = read_lexicon(filename)
    phones = get_phones(lexicon)
    words = get_words(lexicon)
    print(lexicon)
    print(phones)
    print(words)
    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)
    print(lexicon_disambig)
    print("max disambig:", f"#{max_disambig}")

    phones = ["<eps>", "SIL", "SPN"] + phones
    for i in range(max_disambig + 1):
        phones.append(f"#{i}")
    words = ["<eps>"] + words

    phone2id = generate_id_map(phones)
    word2id = generate_id_map(words)

    print(phone2id)
    print(word2id)

    write_mapping("phones.txt", phone2id)
    write_mapping("words.txt", word2id)

    write_lexicon("a.txt", lexicon)
    write_lexicon("a_disambig.txt", lexicon_disambig)

    fsa = lexicon_to_fst(lexicon, phone2id=phone2id, word2id=word2id)
    fsa.labels_sym = k2.SymbolTable.from_file("phones.txt")
    fsa.aux_labels_sym = k2.SymbolTable.from_file("words.txt")
    fsa.draw("L.pdf", title="L")

    fsa_disambig = lexicon_to_fst(lexicon_disambig, phone2id=phone2id, word2id=word2id)
    fsa_disambig.labels_sym = k2.SymbolTable.from_file("phones.txt")
    fsa_disambig.aux_labels_sym = k2.SymbolTable.from_file("words.txt")
    fsa_disambig.draw("L_disambig.pdf", title="L_disambig")


def main():
    filename = generate_lexicon_file()
    test_read_lexicon(filename)
    os.remove(filename)


if __name__ == "__main__":
    main()
