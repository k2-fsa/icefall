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
This script checks that there are no OOV tokens in the BPE-based lexicon.

Usage example:

    python3 ./local/validate_bpe_lexicon.py \
            --lexicon /path/to/lexicon.txt \
            --bpe-model /path/to/bpe.model
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm

from icefall.lexicon import read_lexicon

# Map word to word pieces
Lexicon = List[Tuple[str, List[str]]]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lexicon",
        required=True,
        type=Path,
        help="Path to lexicon.txt",
    )

    parser.add_argument(
        "--bpe-model",
        required=True,
        type=Path,
        help="Path to bpe.model",
    )

    return parser.parse_args()


def main():
    args = get_args()
    assert args.lexicon.is_file(), args.lexicon
    assert args.bpe_model.is_file(), args.bpe_model

    lexicon = read_lexicon(args.lexicon)

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    word_pieces = set(sp.id_to_piece(list(range(sp.vocab_size()))))
    for word, pieces in lexicon:
        for p in pieces:
            if p not in word_pieces:
                raise ValueError(f"The word {word} contains an OOV token {p}")


if __name__ == "__main__":
    main()
