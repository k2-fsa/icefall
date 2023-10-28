#!/usr/bin/env python3
# Copyright           2023  Xiaomi Corp.        (authors: Zengwei Yao)
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
This script split the LJSpeech dataset cuts into three sets:
  - training, 12500
  - validation, 100
  - test, 500
The numbers are from https://arxiv.org/pdf/2106.06103.pdf

Usage example:
    python3 ./local/split_subsets.py ./data/spectrogram
"""

import argparse
import logging
import random
from pathlib import Path

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest_dir",
        type=Path,
        default=Path("data/spectrogram"),
        help="Path to the manifest file",
    )

    return parser.parse_args()


def main():
    args = get_args()

    manifest_dir = Path(args.manifest_dir)
    prefix = "ljspeech"
    suffix = "jsonl.gz"
    # all_cuts = load_manifest_lazy(manifest_dir / f"{prefix}_cuts_all.{suffix}")
    all_cuts = load_manifest_lazy(manifest_dir / f"{prefix}_cuts_all_phonemized.{suffix}")

    cut_ids = list(all_cuts.ids)
    random.shuffle(cut_ids)

    train_cuts = all_cuts.subset(cut_ids=cut_ids[:12500])
    valid_cuts = all_cuts.subset(cut_ids=cut_ids[12500:12500 + 100])
    test_cuts = all_cuts.subset(cut_ids=cut_ids[12500 + 100:])
    assert len(train_cuts) == 12500, "expected 12500 cuts for training but got len(train_cuts)"
    assert len(valid_cuts) == 100, "expected 100 cuts but for validation but got len(valid_cuts)"
    assert len(test_cuts) == 500, "expected 500 cuts for test but got len(test_cuts)"

    train_cuts.to_file(manifest_dir / f"{prefix}_cuts_train.{suffix}")
    valid_cuts.to_file(manifest_dir / f"{prefix}_cuts_valid.{suffix}")
    test_cuts.to_file(manifest_dir / f"{prefix}_cuts_test.{suffix}")

    logging.info("Splitted into three sets: training (12500), validation (100), and test (500)")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
