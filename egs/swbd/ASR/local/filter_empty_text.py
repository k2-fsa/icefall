#!/usr/bin/env python3
# Copyright    2023  The Chinese University of Hong Kong (author: Zengrui Jin)
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
import logging
from typing import List


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kaldi-data-dir",
        type=Path,
        required=True,
        help="Path to the kaldi data dir",
    )

    return parser.parse_args()


def load_segments(path: Path):
    segments = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            utt_id, rec_id, start, end = line.split()
            segments[utt_id] = line
    return segments


def filter_text(path: Path):
    with open(path, "r") as f:
        lines = f.readlines()
        return list(filter(lambda x: len(x.strip().split()) > 1, lines))


def write_segments(path: Path, texts: List[str]):
    with open(path, "w") as f:
        f.writelines(texts)


def main():
    args = get_args()
    orig_text_dict = filter_text(args.kaldi_data_dir / "text")
    write_segments(args.kaldi_data_dir / "text", orig_text_dict)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()

    logging.info("Empty lines filtered")
