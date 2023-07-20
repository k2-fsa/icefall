#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Zengrui Jin)
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

import argparse
from pathlib import Path

from tqdm.auto import tqdm
from icefall.utils import tokenize_by_CJK_char


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Output directory.
        The generated transcript_chars.txt is saved to this directory.
        """,
    )

    parser.add_argument(
        "--text",
        type=str,
        help="WenetSpeech training transcript.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    text = Path(args.text)

    assert lang_dir.exists() and text.exists(), f"{lang_dir} or {text} does not exist!"

    transcript_path = lang_dir / "transcript_chars.txt"

    with open(text, "r", encoding="utf-8") as fin:
        text_lines = fin.readlines()
    tokenized_lines = []
    for line in tqdm(text_lines, desc="Tokenizing training transcript"):
        tokenized_lines.append(f"{tokenize_by_CJK_char(line)}\n")
    with open(transcript_path, "w+", encoding="utf-8") as fout:
        fout.writelines(tokenized_lines)


if __name__ == "__main__":
    main()
