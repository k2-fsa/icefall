#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This file generates the manifest and computes the fbank features for AudioSet
dataset. The generated manifests and features are stored in data/fbank.
"""

import argparse

import lhotse
from lhotse import load_manifest


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-manifest", type=str, default="data/fbank/cuts_audioset_full.jsonl.gz"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    return parser


def main():
    # Reference: https://github.com/YuanGongND/ast/blob/master/egs/audioset/gen_weight_file.py
    parser = get_parser()
    args = parser.parse_args()

    cuts = load_manifest(args.input_manifest)

    print(f"A total of {len(cuts)} cuts.")

    label_count = [0] * 527  # a total of 527 classes
    for c in cuts:
        audio_event = c.supervisions[0].audio_event
        labels = list(map(int, audio_event.split(";")))
        for label in labels:
            label_count[label] += 1

    with open(args.output, "w") as f:
        for c in cuts:
            audio_event = c.supervisions[0].audio_event
            labels = list(map(int, audio_event.split(";")))
            weight = 0
            for label in labels:
                weight += 1000 / (label_count[label] + 0.01)
            f.write(f"{c.id} {weight}\n")


if __name__ == "__main__":
    main()
