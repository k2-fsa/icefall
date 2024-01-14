#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (author: Mingshuang Luo)
# Copyright    2022  Behavox LLC.        (author: Daniil Kulko)
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
This script takes input text file and removes all words
that iclude any character out of English alphabet.

"""
import argparse
import logging
import re
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-text-path",
        type=str,
        help="Input text file path.",
    )
    parser.add_argument(
        "--output-text-path",
        type=str,
        help="Output text file path.",
    )

    return parser.parse_args()


def prepare_transcripts(input_text_path: Path, output_text_path: Path) -> None:
    """
    Args:
      input_text_path:
        The input data text file path, e.g., data/lang/train_orig.txt.
      output_text_path:
        The output data text file path, e.g., data/lang/train.txt.

    Return:
      Saved text file in output_text_path.
    """

    foreign_chr_check = re.compile(r"[^a-z']")

    logging.info(f"Loading {input_text_path.name}")
    with open(input_text_path, "r", encoding="utf8") as f:
        texts = {t.rstrip("\n") for t in f}

    texts = {
        " ".join([w for w in t.split() if foreign_chr_check.search(w) is None])
        for t in texts
    }

    with open(output_text_path, "w+", encoding="utf8") as f:
        for t in texts:
            f.write(f"{t}\n")


def main() -> None:
    args = get_args()
    input_text_path = Path(args.input_text_path)
    output_text_path = Path(args.output_text_path)

    logging.info(f"Generating {output_text_path.name}")
    prepare_transcripts(input_text_path, output_text_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
