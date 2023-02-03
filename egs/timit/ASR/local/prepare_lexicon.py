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
This script takes as input supervisions json dir "data/manifests"
consisting of supervisions_TRAIN.json and does the following:

1. Generate lexicon.txt.

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
      The lexicon.txt file and the train.text in lang_dir.
    """
    import gzip

    phones = set()

    supervisions_train = Path(manifests_dir) / "timit_supervisions_TRAIN.jsonl.gz"
    lexicon = Path(lang_dir) / "lexicon.txt"

    logging.info(f"Loading {supervisions_train}!")
    with gzip.open(supervisions_train, "r") as load_f:
        for line in load_f.readlines():
            load_dict = json.loads(line)
            text = load_dict["text"]
            # list the phone units and filter the empty item
            phones_list = list(filter(None, text.split()))

            for phone in phones_list:
                if phone not in phones:
                    phones.add(phone)

    with open(lexicon, "w") as f:
        for phone in sorted(phones):
            f.write(phone + "  " + phone)
            f.write("\n")
        f.write("<UNK>  <UNK>")
        f.write("\n")


def main():
    args = get_args()
    manifests_dir = Path(args.manifests_dir)
    lang_dir = Path(args.lang_dir)

    logging.info("Generating lexicon.txt")
    prepare_lexicon(manifests_dir, lang_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
