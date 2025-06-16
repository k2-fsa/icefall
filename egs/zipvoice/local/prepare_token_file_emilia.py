#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                       Wei Kang)
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
from typing import List

from piper_phonemize import get_espeak_map
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens_emilia.txt"),
        help="Path to the dict that maps the text tokens to IDs",
    )

    parser.add_argument(
        "--pinyin",
        type=Path,
        default=Path("local/pinyin.txt"),
        help="Path to the all unique pinyin",
    )

    return parser.parse_args()


def get_pinyin_tokens(pinyin: Path) -> List[str]:
    phones = set()
    with open(pinyin, "r") as f:
        for line in f:
            x = line.strip()
            initial = to_initials(x, strict=False)
            # don't want to share tokens with espeak tokens, so use tone3 style
            finals = to_finals_tone3(x, strict=False, neutral_tone_with_five=True)
            if initial != "":
                # don't want to share tokens with espeak tokens, so add a '0' after each initial
                phones.add(initial + "0")
            if finals != "":
                phones.add(finals)
    return sorted(phones)


def get_token2id(args):
    """Get a dict that maps token to IDs, and save it to the given filename."""
    all_tokens = get_espeak_map()  # token: [token_id]
    all_tokens = {token: token_id[0] for token, token_id in all_tokens.items()}
    # sort by token_id
    all_tokens = sorted(all_tokens.items(), key=lambda x: x[1])

    all_pinyin = get_pinyin_tokens(args.pinyin)
    with open(args.tokens, "w", encoding="utf-8") as f:
        for token, token_id in all_tokens:
            f.write(f"{token} {token_id}\n")
        num_espeak_tokens = len(all_tokens)
        for i, pinyin in enumerate(all_pinyin):
            f.write(f"{pinyin} {num_espeak_tokens + i}\n")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    get_token2id(args)
