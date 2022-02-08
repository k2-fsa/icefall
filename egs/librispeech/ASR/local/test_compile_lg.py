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

from pathlib import Path
from typing import List

import k2
import sentencepiece as spm
import torch

lang_dir = Path("./data/lang_bpe_500")


def get_word_ids(word_table: k2.SymbolTable, s: str) -> List[int]:
    """
    Args:
      word_table:
        Word symbol table.
      s:
        A string consisting of space(s) separated words.
    Returns:
      Return a list of word IDs.
    """
    ans = []
    for w in s.split():
        ans.append(word_table[w])
    return ans


def main():
    assert lang_dir.exists(), f"{lang_dir} does not exist!"
    LG = k2.Fsa.from_dict(torch.load(f"{lang_dir}/LG.pt", map_location="cpu"))

    sp = spm.SentencePieceProcessor()
    sp.load(f"{lang_dir}/bpe.model")

    word_table = k2.SymbolTable.from_file(f"{lang_dir}/words.txt")
    s = "HELLO WORLD"
    token_ids = sp.encode(s)

    token_fsa = k2.linear_fsa(token_ids)

    fsa = k2.intersect(LG, token_fsa)
    fsa = k2.connect(fsa)
    print(k2.to_dot(fsa))
    print(fsa.properties_str)
    print(LG.properties_str)
    # You can use https://dreampuf.github.io/GraphvizOnline/
    # to visualize the output.
    #
    # You can see that the resulting fsa is not deterministic
    # Note: LG is non-deterministic
    #
    # See https://shorturl.at/uIL69
    # for visualization of the above fsa.


if __name__ == "__main__":
    main()
