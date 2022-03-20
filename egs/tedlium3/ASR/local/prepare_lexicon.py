#!/usr/bin/env python3
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
This script takes as input supervisions json dir "data/manifests"
consisting of supervisions_train.json and does the following:

1. Generate lexicon_words.txt.

"""
import argparse
import json
import logging
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests-dir",
        type=str,
        help="""Input directory.
        """,
    )
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Output directory.
        """,
    )

    return parser.parse_args()


def prepare_lexicon(manifests_dir: str, lang_dir: str):
    """
    Args:
      manifests_dir:
        The manifests directory, e.g., data/manifests.
      lang_dir:
        The language directory, e.g., data/lang_phone.

    Return:
      The lexicon_words.txt file.
    """
    words = set()

    supervisions_train = Path(manifests_dir) / "supervisions_train.json"
    lexicon = Path(lang_dir) / "lexicon_words.txt"

    logging.info(f"Loading {supervisions_train}!")
    with open(supervisions_train, "r") as load_f:
        load_dicts = json.load(load_f)
        for load_dict in load_dicts:
            text = load_dict["text"]
            # list the words units and filter the empty item
            words_list = list(filter(None, text.split()))

            for word in words_list:
                if word not in words and word != "<unk>":
                    words.add(word)

    with open(lexicon, "w") as f:
        for word in sorted(words):
            f.write(word + "  " + word)
            f.write("\n")


def main():
    args = get_args()
    manifests_dir = Path(args.manifests_dir)
    lang_dir = Path(args.lang_dir)

    logging.info("Generating lexicon_words.txt")
    prepare_lexicon(manifests_dir, lang_dir)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
