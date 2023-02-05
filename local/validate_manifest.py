#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This script checks the following assumptions of the generated manifest:

- Single supervision per cut
- Supervision time bounds are within cut time bounds

We will add more checks later if needed.

Usage example:

    python3 ./local/validate_manifest.py \
            ./data/fbank/librispeech_cuts_train-clean-100.jsonl.gz

"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest file",
    )

    return parser.parse_args()


def validate_one_supervision_per_cut(c: Cut):
    if len(c.supervisions) != 1:
        raise ValueError(f"{c.id} has {len(c.supervisions)} supervisions")


def validate_supervision_and_cut_time_bounds(c: Cut):
    s = c.supervisions[0]
    if s.start < c.start:
        raise ValueError(
            f"{c.id}: Supervision start time {s.start} is less "
            f"than cut start time {c.start}"
        )

    if s.end > c.end:
        raise ValueError(
            f"{c.id}: Supervision end time {s.end} is larger "
            f"than cut end time {c.end}"
        )


def main():
    args = get_args()

    manifest = args.manifest
    logging.info(f"Validating {manifest}")

    assert manifest.is_file(), f"{manifest} does not exist"
    cut_set = load_manifest_lazy(manifest)
    assert isinstance(cut_set, CutSet)

    for c in cut_set:
        validate_one_supervision_per_cut(c)
        validate_supervision_and_cut_time_bounds(c)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
