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
    all_tokens = get_espeak_map()  # token: [token_id]
    all_tokens = {token: token_id[0] for token, token_id in all_tokens.items()}
    # sort by token_id
    all_tokens = sorted(all_tokens.items(), key=lambda x: x[1])

    with open(filename, "w", encoding="utf-8") as f:
        for token, token_id in all_tokens:
            f.write(f"{token} {token_id}\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    out_file = Path(args.tokens)
    get_token2id(out_file)
