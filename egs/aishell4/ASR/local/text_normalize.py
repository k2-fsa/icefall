#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
This script takes as input "text_full", which includes three transcript files
(train_S, train_M and train_L) for AISHELL4:
    - text_full
and generates the output file text_normalize which is implemented
to normalize text:
    - text
"""


import argparse

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Normalizing for text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/lang_char/text_full",
        type=str,
        help="the input text files for AISHELL4",
    )
    parser.add_argument(
        "--output",
        default="data/lang_char/text",
        type=str,
        help="the text implemented with normalizer for AISHELL4",
    )

    return parser


def text_normalize(str_line: str):
    line = str_line.strip().rstrip("\n")
    line = line.replace(" ", "")
    line = line.replace("<sil>", "")
    line = line.replace("<%>", "")
    line = line.replace("<->", "")
    line = line.replace("<$>", "")
    line = line.replace("<#>", "")
    line = line.replace("<_>", "")
    line = line.replace("<space>", "")
    line = line.replace("`", "")
    line = line.replace("&", "")
    line = line.replace(",", "")
    line = line.replace("Ａ", "")
    line = line.replace("ａ", "A")
    line = line.replace("ｂ", "B")
    line = line.replace("ｃ", "C")
    line = line.replace("ｋ", "K")
    line = line.replace("ｔ", "T")
    line = line.replace("，", "")
    line = line.replace("丶", "")
    line = line.replace("。", "")
    line = line.replace("、", "")
    line = line.replace("？", "")
    line = line.replace("·", "")
    line = line.replace("*", "")
    line = line.replace("！", "")
    line = line.replace("$", "")
    line = line.replace("+", "")
    line = line.replace("-", "")
    line = line.replace("\\", "")
    line = line.replace("?", "")
    line = line.replace("￥", "")
    line = line.replace("%", "")
    line = line.replace(".", "")
    line = line.replace("<", "")
    line = line.replace("＆", "")
    line = line.upper()

    return line


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()
    new_lines = []
    for i in tqdm(range(len(lines))):
        new_line = text_normalize(lines[i])
        new_lines.append(new_line)

    f_new = open(output_file, "w", encoding="utf-8")
    for line in new_lines:
        f_new.write(line)
        f_new.write("\n")


if __name__ == "__main__":
    main()
