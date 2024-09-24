#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright    2021  Xiaomi Corp.        (authors: Mingshuang Luo)
#              2022  Xiaomi Corp.        (authors: Weiji Zhuang)
#              2024  Xiaomi Corp.        (authors: Zengrui Jin)
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
This script takes as input "text", which refers to the transcript file for
MDCC:
    - text
and generates the output file text_word_segmentation which is implemented
with word segmenting:
    - text_words_segmentation
"""

import argparse
from typing import List

import pycantonese
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Cantonese Word Segmentation for text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default="data/lang_char/text",
        type=str,
        help="the input text file for MDCC",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="data/lang_char/text_words_segmentation",
        type=str,
        help="the text implemented with words segmenting for MDCC",
    )

    return parser


def get_word_segments(lines: List[str]) -> List[str]:
    return [
        " ".join(pycantonese.segment(line)) + "\n"
        for line in tqdm(lines, desc="Segmenting lines")
    ]


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

        new_lines = get_word_segments(lines)

    with open(output_file, "w", encoding="utf-8") as fw:
        fw.writelines(new_lines)


if __name__ == "__main__":
    main()
