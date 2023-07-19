#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest", type=str, help="The original manifest coming from"
    )

    return parser.parse_args()


def main(args):

    logging.info(f"Loading manifest {args.manifest}")
    cuts = load_manifest_lazy(args.manifest)

    all_test_sets = [
        "dev",
        "test-clean",
        "test-other",
    ]

    for test_set in all_test_sets:
        logging.info(f"Processing test set: {test_set}")
        with open(f"data/manifests/{test_set}.txt", "r") as f:
            books = f.read().split("\n")

        # find the cuts belonging to the given books
        selected_cuts = cuts.filter(lambda c: c.text_path.split("/")[-2] in books)
        selected_cuts.describe()

        out_name = f"data/manifests/libriheavy_cuts_{test_set}.jsonl.gz"
        logging.info(f"Saving the cuts contained in the book list to {out_name}")
        selected_cuts.to_file(out_name)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))

    main(args)
