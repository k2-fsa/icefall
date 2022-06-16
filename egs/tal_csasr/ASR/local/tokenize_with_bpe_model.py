#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright    2021 Mobvoi Inc.          (authors: Binbin Zhang)
# Copyright    2022  Xiaomi Corp.        (authors: Mingshuang Luo)
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
This script takes as input text (it includes Chinese and English):
    - text
and generates the text_with_bpe.
    - text_with_bpe
"""


import argparse
import logging
import re

import sentencepiece as spm
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare text_with_bpe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/lang_char/text",
        type=str,
        help="the text includes Chinese and English words",
    )
    parser.add_argument(
        "--output",
        default="data/lang_char/text_with_bpe",
        type=str,
        help="the text_with_bpe tokenized by bpe model",
    )
    parser.add_argument(
        "--bpe-model",
        default="data/lang_char/bpe.model",
        type=str,
        help="the bpe model for processing the English parts",
    )

    return parser


def tokenize_by_bpe_model(sp, txt):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r"([\u4e00-\u9fff])")
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    bpe_model = args.bpe_model

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model)

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()

    logging.info("Starting reading the text")
    new_lines = []
    for i in tqdm(range(len(lines))):
        x = lines[i]
        txt_tokens = tokenize_by_bpe_model(sp, x)
        new_line = " ".join(txt_tokens)
        new_lines.append(new_line)

    logging.info("Starting writing the text_with_bpe")
    f_out = open(output_file, "w", encoding="utf-8")
    for line in new_lines:
        f_out.write(line)
        f_out.write("\n")


if __name__ == "__main__":
    main()
