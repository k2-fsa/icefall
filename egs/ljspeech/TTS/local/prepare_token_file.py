#!/usr/bin/env python3
# Copyright         2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
This file generates the file that maps tokens to IDs.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

from piper_phonemize import get_espeak_map


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )

    return parser.parse_args()


def get_token2id(filename: Path) -> Dict[str, int]:
    """Get a dict that maps token to IDs, and save it to the given filename."""
    extra_tokens = [
        "<blk>",  # 0 for blank
        "<sos>",  # 1 for sos
        "<eos>",  # 2 for eos
        "<unk>",  # 3 for OOV
    ]

    all_tokens = list(get_espeak_map().keys())

    for t in extra_tokens:
        assert t not in all_tokens, t

    all_tokens = extra_tokens + all_tokens

    with open(filename, "w", encoding="utf-8") as f:
        for i, token in enumerate(all_tokens):
            f.write(f"{token} {i}\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    out_file = Path(args.tokens)
    get_token2id(out_file)
