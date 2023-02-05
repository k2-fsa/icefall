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

import sentencepiece as spm
from tqdm import tqdm

#from icefall.utils import tokenize_by_bpe_model

import re
from bpemb import BPEmb
from bpemb.util import sentencepiece_load, load_word2vec_file


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


def tokenize_by_bpe_model(content,bpe_model):
      tokens=[]
      pattern = re.compile(r"([\u4e00-\u9fff])")
      chars = pattern.split(content.lower())
      mix_chars = [w for w in chars if len(w.strip()) > 0]
      for ch_or_w in mix_chars:
        if pattern.fullmatch(ch_or_w) is not None:
           tokens.append(ch_or_w)
        else:
           bpemb_en = BPEmb(lang="zh")
           bpemb_en.spm = sentencepiece_load(bpe_model)
           ch_or_w=bpemb_en.encode(ch_or_w)
           tokens.append(' '.join(ch_or_w))
      return ' '.join(tokens)


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    bpe_model = args.bpe_model
    print('bpe_model',bpe_model)
    
    

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()

    logging.info("Starting reading the text")
    f_out = open(output_file, "w", encoding="utf-8")
    for i in tqdm(range(len(lines))):
        content = lines[i]
        new_line = tokenize_by_bpe_model(content,bpe_model)
        f_out.write(new_line.upper())
        f_out.write("\n")
    f_out.close()

if __name__ == "__main__":
    main()
