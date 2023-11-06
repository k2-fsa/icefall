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
This file reads the texts in given manifest and generates the file that maps tokens to IDs.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict

import g2p_en
import tacotron_cleaner.cleaners
from lhotse import load_manifest


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("data/spectrogram/ljspeech_cuts_train.jsonl.gz"),
        help="Path to the manifest file",
    )

    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/tokens.txt"),
        help="Path to the tokens",
    )

    return parser.parse_args()


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def get_token2id(manifest_file: Path) -> Dict[str, int]:
    """Return a dict that maps token to IDs."""
    extra_tokens = [
        ("<blk>", None),  # 0 for blank
        ("<sos/eos>", None),  # 1 for sos and eos symbols.
        ("<unk>", None),  # 2 for OOV
    ]
    cut_set = load_manifest(manifest_file)
    g2p = g2p_en.G2p()
    counter = Counter()

    for cut in cut_set:
        # Each cut only contain one supervision
        assert len(cut.supervisions) == 1, len(cut.supervisions)
        text = cut.supervisions[0].normalized_text
        # Text normalization
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        # Convert to phonemes
        tokens = g2p(text)
        for t in tokens:
            counter[t] += 1

    # Sort by the number of occurrences in descending order
    tokens_and_counts = sorted(counter.items(), key=lambda x: -x[1])

    tokens_and_counts = extra_tokens + tokens_and_counts

    token2id: Dict[str, int] = {token: i for i, (token, _) in enumerate(tokens_and_counts)}

    return token2id


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    manifest_file = Path(args.manifest_file)
    out_file = Path(args.tokens)

    token2id = get_token2id(manifest_file)
    write_mapping(out_file, token2id)
