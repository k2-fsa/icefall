#!/usr/bin/env python3
# Copyright    2022-2024  Xiaomi Corp.             (authors: Fangjun Kuang,
#                                                            Zengwei Yao,)
#              2024       The Chinese Univ. of HK  (authors: Zengrui Jin)
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

We will add more checks later if needed.

Usage example:

    python3 ./local/validate_manifest.py \
            ./data/fbank/libritts_cuts_train-all-shuf.jsonl.gz

"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest
from lhotse.dataset.speech_recognition import validate_for_asr


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest file",
    )

    return parser.parse_args()


def main():
    args = get_args()

    manifest = args.manifest
    logging.info(f"Validating {manifest}")

    assert manifest.is_file(), f"{manifest} does not exist"
    cut_set = load_manifest(manifest)
    assert isinstance(cut_set, CutSet)

    validate_for_asr(cut_set)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
