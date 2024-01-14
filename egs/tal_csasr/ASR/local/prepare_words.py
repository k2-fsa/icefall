#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright    2021  Xiaomi Corp.        (authors: Mingshuang Luo)
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
This script takes as input words.txt without ids:
    - words_no_ids.txt
and generates the new words.txt with related ids.
    - words.txt
"""


import argparse
import logging

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare words.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="data/lang_char/words_no_ids.txt",
        type=str,
        help="the words file without ids for WenetSpeech",
    )
    parser.add_argument(
        "--output",
        default="data/lang_char/words.txt",
        type=str,
        help="the words file with ids for WenetSpeech",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()
    new_lines = []
    add_words = ["<eps> 0", "!SIL 1", "<SPOKEN_NOISE> 2", "<UNK> 3"]
    new_lines.extend(add_words)

    logging.info("Starting reading the input file")
    for i in tqdm(range(len(lines))):
        x = lines[i]
        idx = 4 + i
        new_line = str(x.strip("\n")) + " " + str(idx)
        new_lines.append(new_line)

    logging.info("Starting writing the words.txt")
    f_out = open(output_file, "w", encoding="utf-8")
    for line in new_lines:
        f_out.write(line)
        f_out.write("\n")


if __name__ == "__main__":
    main()
