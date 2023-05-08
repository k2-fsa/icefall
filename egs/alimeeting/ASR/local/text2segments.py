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
This script takes as input "text", which refers to the transcript file for
WenetSpeech:
    - text
and generates the output file text_word_segmentation which is implemented
with word segmenting:
    - text_words_segmentation
"""


import argparse

import jieba
import paddle
from tqdm import tqdm

paddle.enable_static()
jieba.enable_paddle()


def get_parser():
    parser = argparse.ArgumentParser(
        description="Chinese Word Segmentation for text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        default="data/lang_char/text",
        type=str,
        help="the input text file for WenetSpeech",
    )
    parser.add_argument(
        "--output-file",
        default="data/lang_char/text_words_segmentation",
        type=str,
        help="the text implemented with words segmenting for WenetSpeech",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    f = open(input_file, "r", encoding="utf-8")
    lines = f.readlines()
    new_lines = []
    for i in tqdm(range(len(lines))):
        x = lines[i].rstrip()
        seg_list = jieba.cut(x, use_paddle=True)
        new_line = " ".join(seg_list)
        new_lines.append(new_line)

    f_new = open(output_file, "w", encoding="utf-8")
    for line in new_lines:
        f_new.write(line)
        f_new.write("\n")


if __name__ == "__main__":
    main()
