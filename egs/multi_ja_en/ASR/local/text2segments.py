#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright    2021  Xiaomi Corp.        (authors: Mingshuang Luo)
#              2022  Xiaomi Corp.        (authors: Weiji Zhuang)
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
This script takes as input "text", which refers to the transcript file:
    - text
and generates the output file with word segmentation implemented using MeCab:
    - text_words_segmentation
"""

import argparse
from multiprocessing import Pool

import MeCab
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Japanese Word Segmentation for text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-process",
        "-n",
        default=20,
        type=int,
        help="the number of processes",
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default="data/lang_char/text",
        type=str,
        help="the input text file",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="data/lang_char/text_words_segmentation",
        type=str,
        help="the text implemented with word segmentation using MeCab",
    )

    return parser


def cut(lines):
    if lines is not None:
        mecab = MeCab.Tagger("-Owakati")  # Use '-Owakati' option for word segmentation
        segmented_line = mecab.parse(lines).strip()
        return segmented_line.split()  # Return as a list of words
    else:
        return None


def main():
    parser = get_parser()
    args = parser.parse_args()

    num_process = args.num_process
    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    with Pool(processes=num_process) as p:
        new_lines = list(tqdm(p.imap(cut, lines), total=len(lines)))

    with open(output_file, "w", encoding="utf-8") as fw:
        for line in new_lines:
            fw.write(" ".join(line) + "\n")


if __name__ == "__main__":
    main()
