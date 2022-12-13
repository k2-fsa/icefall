#!/usr/bin/env python3
# Copyright    2022  Behavox LLC.        (authors: Daniil Kulko)
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
This script takes as input supervisions json dir "data/manifests"
consisting of tedlium_supervisions_train.json and does the following:

1. Generate words.txt.

"""
import argparse
import logging
import re
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="Output directory.",
    )

    return parser.parse_args()


def prepare_words(lang_dir: str) -> None:
    """
    Args:
      lang_dir:
        The language directory, e.g., data/lang.

    Return:
      The words.txt file.
    """

    words_orig_path = Path(lang_dir) / "words_orig.txt"
    words_path = Path(lang_dir) / "words.txt"

    foreign_chr_check = re.compile(r"[^a-z']")

    logging.info(f"Loading {words_orig_path.name}")
    with open(words_orig_path, "r", encoding="utf8") as f:
        words = {w for w_compl in f for w in w_compl.strip("-\n").split("_")}
    words = {w for w in words if foreign_chr_check.search(w) is None and w != ""}
    words.add("<unk>")
    words = ["<eps>", "!SIL"] + sorted(words) + ["#0", "<s>", "</s>"]

    with open(words_path, "w+", encoding="utf8") as f:
        for idx, word in enumerate(words):
            f.write(f"{word} {idx}\n")


def main() -> None:
    args = get_args()
    lang_dir = Path(args.lang_dir)

    logging.info("Generating words.txt")
    prepare_words(lang_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
