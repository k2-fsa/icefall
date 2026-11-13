#!/usr/bin/env python3
# Copyright 2026 Shanghai Jiao Tong University (Author: Yifan Yang)
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
"""Prepare standard LibriSpeech/MUSAN cuts for online 128-bin FBank extraction.

First run the shared ASR preparation stages 0--2. This script creates cuts from
the resulting recording/supervision manifests, without computing 80-bin features
or filtering evaluation utterances. Existing output cuts are preserved.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool

LIBRISPEECH_SPLITS = (
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
)


def prepare_librispeech(manifest_dir, output_dir, splits, perturb_speed):
    manifests = read_manifests_if_cached(
        dataset_parts=splits,
        output_dir=manifest_dir,
        prefix="librispeech",
        suffix="jsonl.gz",
    )
    if manifests is None or any(split not in manifests for split in splits):
        raise FileNotFoundError(
            f"LibriSpeech manifests missing from {manifest_dir}; run prepare.sh stages 0--2"
        )
    for split in splits:
        output = output_dir / f"librispeech_cuts_{split}.jsonl.gz"
        if output.exists():
            logging.info("Keeping existing cuts: %s", output)
            continue
        cuts = CutSet.from_manifests(**manifests[split])
        if split.startswith("train-") and perturb_speed:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        cuts.to_file(output)
        logging.info("Wrote %s", output)


def prepare_musan(manifest_dir, output_dir):
    output = output_dir / "musan_cuts.jsonl.gz"
    if output.exists():
        logging.info("Keeping existing MUSAN cuts: %s", output)
        return
    parts = ("music", "speech", "noise")
    manifests = read_manifests_if_cached(
        dataset_parts=parts,
        output_dir=manifest_dir,
        prefix="musan",
        suffix="jsonl.gz",
    )
    if manifests is None or any(part not in manifests for part in parts):
        raise FileNotFoundError(
            f"MUSAN manifests missing from {manifest_dir}; run prepare.sh stage 2"
        )
    recordings = combine(manifests[part]["recordings"] for part in parts)
    cuts = (
        CutSet.from_manifests(recordings=recordings)
        .cut_into_windows(10.0)
        .filter(lambda c: c.duration > 5.0)
    )
    cuts.to_file(output)
    logging.info("Wrote %s", output)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fbank"))
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=LIBRISPEECH_SPLITS,
        default=list(LIBRISPEECH_SPLITS),
    )
    parser.add_argument("--perturb-speed", type=str2bool, default=True)
    parser.add_argument("--prepare-musan", type=str2bool, default=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_librispeech(
        args.manifest_dir, args.output_dir, args.dataset, args.perturb_speed
    )
    if args.prepare_musan:
        prepare_musan(args.manifest_dir, args.output_dir)


if __name__ == "__main__":
    main()
