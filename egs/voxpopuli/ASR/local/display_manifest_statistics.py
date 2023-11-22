#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2023  Brno University of Technology  (authors: Karel Veselý)
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
This file displays duration statistics of utterances in a manifest.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.

Usage example:
    python3 ./local/display_manifest_statistics.py data/fbank/*_cuts*.jsonl.gz

See the function `remove_short_and_long_utt()` in transducer/train.py
for usage.

"""

import argparse

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser("Compute statistics for 'cuts' .jsonl.gz")

    parser.add_argument(
        "filename",
        help="data/fbank/imported_cuts_bison-train_trim.jsonl.gz",
    )

    return parser.parse_args()


def main():
    args = get_args()

    cuts = load_manifest_lazy(args.filename)
    cuts.describe()


if __name__ == "__main__":
    main()
