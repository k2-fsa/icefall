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
from symbols import symbols


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )

    return parser.parse_args()


def main():
    args = get_args()
    tokens = Path(args.tokens)

    with open(tokens, "w", encoding="utf-8") as f:
        for token_id, token in enumerate(symbols):
            f.write(f"{token} {token_id}\n")


if __name__ == "__main__":
    main()
