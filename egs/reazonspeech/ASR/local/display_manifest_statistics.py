#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  The University of Electro-Communications (author: Teo Wen Shen)  # noqa
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

import argparse
from pathlib import Path

from lhotse import CutSet, load_manifest

ARGPARSE_DESCRIPTION = """
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

See the function `remove_short_and_long_utt()` in
pruned_transducer_stateless5/train.py for usage.
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest-dir", type=Path, help="Path to cutset manifests")

    return parser.parse_args()


def main():
    args = get_parser()

    for part in ["train", "dev"]:
        path = args.manifest_dir / f"reazonspeech_cuts_{part}.jsonl.gz"
        cuts: CutSet = load_manifest(path)

        print("\n---------------------------------\n")
        print(path.name + ":")
        cuts.describe()


if __name__ == "__main__":
    main()
