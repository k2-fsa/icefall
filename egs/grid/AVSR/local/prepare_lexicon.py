#!/usr/bin/env python3
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
This script takes as input dir "download/GRID/GRID_align_txt"
consisting of all samples' text files and does the following:

1. Generate lexicon.txt.

2. Generate train.text.
"""
import argparse
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-txt",
        type=str,
        help="""The file listing training samples.
        """,
    )
    parser.add_argument(
        "--align-dir",
        type=str,
        help="""The directory including training samples'
        text files.
        """,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Output directory.
        """,
    )

    return parser.parse_args()


def prepare_lexicon(
    train_samples_txt: str, train_align_dir: str, lang_dir: str
):
    """
    Args:
      train_samples_txt:
        The file listing training samples, e.g., download/GRID/unseen_train.txt.
      train_align_dir:
        The directory including training samples' text files,
        e.g., download/GRID/GRID_align_txt.
      lang_dir:
        Output directory, e.g., data/lang_character
    Return:
      The lexicon.txt file and the train.text in lang_dir.
    """
    words = set()

    train_text = Path(lang_dir) / "train.text"
    lexicon = Path(lang_dir) / "lexicon.txt"

    if train_text.exists() is False:
        texts = []
        train_samples_txts = []
        with open(train_samples_txt, "r") as f:
            train_samples_txts = [line.strip() for line in f.readlines()]

        for sample_txt in train_samples_txts:
            anno = sample_txt.replace("video/mpg_6000", "align") + ".align"
            anno = Path(train_align_dir) / anno
            with open(anno, "r") as f:
                lines = [line.strip().split(" ") for line in f.readlines()]
                txt = [line[2] for line in lines]
                txt = list(
                    filter(lambda s: not s.upper() in ["SIL", "SP"], txt)
                )
                txt = " ".join(txt)
                texts.append(txt.upper())

        with open(train_text, "w") as f:
            for txt in texts:
                f.write(txt)
                f.write("\n")

    with open(train_text, "r") as load_f:
        lines = load_f.readlines()
        for line in lines:
            words_list = list(filter(None, line.rstrip("\n").split(" ")))
            for word in words_list:
                if word not in words:
                    words.add(word)

    with open(lexicon, "w") as f:
        for word in words:
            chars = list(word)
            char_str = " ".join(chars)
            f.write((word + "  " + char_str).upper())
            f.write("\n")
        f.write("<UNK>  <UNK>")
        f.write("\n")


def main():
    args = get_args()
    train_samples_txt = Path(args.samples_txt)
    train_align_dir = Path(args.align_dir)
    lang_dir = Path(args.lang_dir)

    logging.info("Generating lexicon.txt and train.text")

    prepare_lexicon(train_samples_txt, train_align_dir, lang_dir)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
