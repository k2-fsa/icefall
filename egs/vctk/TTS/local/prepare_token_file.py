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
This file reads the texts in given manifest and generate the file that maps tokens to IDs.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict

import g2p_en
import tacotron_cleaner.cleaners
from lhotse import load_manifest
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("data/spectrogram/vctk_cuts_all.jsonl.gz"),
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
    extra_tokens = {
        "<blk>": 0,  # blank
        "<sos/eos>": 1,  # sos and eos symbols.
        "<unk>": 2,  # OOV
    }
    cut_set = load_manifest(manifest_file)
    g2p = g2p_en.G2p()
    counter = Counter()

    for cut in tqdm(cut_set):
        # Each cut only contain one supervision
        assert len(cut.supervisions) == 1, len(cut.supervisions)
        text = cut.supervisions[0].text
        # Text normalization
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        # Convert to phonemes
        tokens = g2p(text)
        for t in tokens:
            counter[t] += 1

    # Sort by the number of occurrences in descending order
    tokens_and_counts = sorted(counter.items(), key=lambda x: -x[1])

    for token, idx in extra_tokens.items():
        tokens_and_counts.insert(idx, (token, None))

    token2id: Dict[str, int] = {
        token: i for i, (token, count) in enumerate(tokens_and_counts)
    }
    return token2id


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    manifest_file = Path(args.manifest_file)
    out_file = Path(args.tokens)

    token2id = get_token2id(manifest_file)
    write_mapping(out_file, token2id)
