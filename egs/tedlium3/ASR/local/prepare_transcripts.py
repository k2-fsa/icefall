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
consisting of supervisions_train.json and does the following:

1. Generate train.text.

"""
import argparse
import logging
from pathlib import Path

import lhotse


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


def prepare_transcripts(manifests_dir: str, lang_dir: str):
    """
    Args:
      manifests_dir:
        The manifests directory, e.g., data/manifests.
      lang_dir:
        The language directory, e.g., data/lang_phone.

    Return:
      The train.text in lang_dir.
    """
    texts = []

    train_text = Path(lang_dir) / "train.text"
    sups = lhotse.load_manifest(f"{manifests_dir}/tedlium_supervisions_train.jsonl.gz")
    for s in sups:
        texts.append(s.text)

    with open(train_text, "w") as f:
        for text in texts:
            f.write(text)
            f.write("\n")


def main():
    args = get_args()
    manifests_dir = Path(args.manifests_dir)
    lang_dir = Path(args.lang_dir)

    logging.info("Generating train.text")
    prepare_transcripts(manifests_dir, lang_dir)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
