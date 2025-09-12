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
This file reads the texts in given manifest and save the new cuts with phoneme tokens.
"""

import logging
from pathlib import Path

try:
    import tacotron_cleaner.cleaners
except ModuleNotFoundError as ex:
    raise RuntimeError(f"{ex}\nPlease run\n  pip install espnet_tts_frontend\n")

import argparse

from lhotse import CutSet, load_manifest
from piper_phonemize import phonemize_espeak


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--in-out-dir",
        type=Path,
        required=True,
        help="Input and output directory",
    )

    return parser


def prepare_tokens_ljspeech(in_out_dir):
    prefix = "ljspeech"
    suffix = "jsonl.gz"
    partition = "all"

    cut_set = load_manifest(in_out_dir / f"{prefix}_cuts_{partition}.{suffix}")

    new_cuts = []
    for cut in cut_set:
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].normalized_text
        # Text normalization
        text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
        # Convert to phonemes
        tokens_list = phonemize_espeak(text, "en-us")
        tokens = []
        for t in tokens_list:
            tokens.extend(t)
        cut.tokens = tokens
        new_cuts.append(cut)

    new_cut_set = CutSet.from_cuts(new_cuts)
    new_cut_set.to_file(in_out_dir / f"{prefix}_cuts_with_tokens_{partition}.{suffix}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_parser().parse_args()

    prepare_tokens_ljspeech(args.in_out_dir)
