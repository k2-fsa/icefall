#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Zengrui Jin)
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
This script takes a text file "data/lang_char/text" as input, the file consist of
lines each containing a transcript, applies text norm and generates the following
files in the directory "data/lang_char":
    - transcript_words.txt
    - words.txt
    - words_no_ids.txt
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List

import pycantonese
from preprocess_commonvoice import normalize_text
from tqdm.auto import tqdm

from icefall.utils import is_cjk, tokenize_by_CJK_char


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare char lexicon",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default="data/yue/lang_char/text",
        type=str,
        help="The input text file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/yue/lang_char/",
        type=str,
        help="The output directory",
    )
    parser.add_argument(
        "--lang",
        "-l",
        default="yue",
        type=str,
        help="The language",
    )
    return parser


def get_word_segments(lines: List[str]) -> List[str]:
    # the current pycantonese segmenter does not handle the case when the input
    # is code switching, so we need to handle it separately

    new_lines = []

    for line in tqdm(lines, desc="Segmenting lines"):
        try:
            if is_cs(line):  # code switching
                segments = []
                curr_str = ""
                for segment in tokenize_by_CJK_char(line).split(" "):
                    if segment.strip() == "":
                        continue
                    try:
                        if not is_cjk(segment[0]):  # en segment
                            if curr_str:
                                segments.extend(pycantonese.segment(curr_str))
                                curr_str = ""
                            segments.append(segment)
                        else:  # zh segment
                            curr_str += segment
                            # segments.extend(pycantonese.segment(segment))
                    except Exception as e:
                        logging.error(f"Failed to process segment: {segment}")
                        raise
                if curr_str:  # process the last segment
                    segments.extend(pycantonese.segment(curr_str))
                new_lines.append(" ".join(segments) + "\n")
            else:  # not code switching
                new_lines.append(" ".join(pycantonese.segment(line)) + "\n")
        except Exception as e:
            logging.error(f"Failed to process line: {line}")
            raise e
    return new_lines


def get_words(lines: List[str]) -> List[str]:
    words = set()
    for line in tqdm(lines, desc="Getting words"):
        words.update(line.strip().split(" "))
    return list(words)


def is_cs(line: str) -> bool:
    english_markers = r"[a-zA-Z]+"
    return bool(re.search(english_markers, line))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    lang = args.lang

    assert input_file.is_file(), f"{input_file} does not exist"
    assert output_dir.is_dir(), f"{output_dir} does not exist"

    lines = input_file.read_text(encoding="utf-8").strip().split("\n")
    norm_lines = [normalize_text(line, lang) for line in lines]

    text_words_segments = get_word_segments(norm_lines)
    with open(output_dir / "transcript_words.txt", "w", encoding="utf-8") as f:
        f.writelines(text_words_segments)

    words = get_words(text_words_segments)[1:]  # remove "\n" from words
    with open(output_dir / "words_no_ids.txt", "w", encoding="utf-8") as f:
        f.writelines([word + "\n" for word in sorted(words)])

    words = (
        ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>"]
        + sorted(words)
        + ["#0", "<s>", "<\s>"]
    )

    with open(output_dir / "words.txt", "w", encoding="utf-8") as f:
        f.writelines([f"{word} {i}\n" for i, word in enumerate(words)])
