#!/usr/bin/env python3
# Copyright    2023  Brno University of Technology  (authors: Karel Veselý)
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
Print the text contained in `supervisions.jsonl.gz` or `cuts.jsonl.gz`.

Usage example:
    python3 ./local/text_from_manifest.py \
        data/manifests/voxpopuli-asr-en_supervisions_dev.jsonl.gz
"""

import argparse
import gzip
import json


def get_args():
    parser = argparse.ArgumentParser(
        "Read the raw text from the 'supervisions.jsonl.gz'"
    )
    parser.add_argument("filename", help="supervisions.jsonl.gz")
    return parser.parse_args()


def main():
    args = get_args()

    with gzip.open(args.filename, mode="r") as fd:
        for line in fd:
            js = json.loads(line)
            if "text" in js:
                print(js["text"])  # supervisions.jsonl.gz
            elif "supervisions" in js:
                for s in js["supervisions"]:
                    print(s["text"])  # cuts.jsonl.gz
            else:
                raise Exception(f"Unknown jsonl format of {args.filename}")


if __name__ == "__main__":
    main()
