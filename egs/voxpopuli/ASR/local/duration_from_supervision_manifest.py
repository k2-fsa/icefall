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
This script computes durations of datasets from
the SupervisionSet manifests.

Usage example:

  python3 ./local/duration_from_supervision_manifest.py \
    data/manifest/*_superivions*.jsonl.gz
"""

import argparse
import gzip
import json
import logging
import re
import sys


def get_args():
    parser = argparse.ArgumentParser(
        "Read the raw text from the 'supervisions.jsonl.gz'"
    )

    parser.add_argument(
        "filename",
        help="supervisions.jsonl.gz",
        nargs="+",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))

    total_duration = 0.0
    total_n_utts = 0

    for fname in args.filename:
        if fname == "-":
            fd = sys.stdin
        elif re.match(r".*\.jsonl\.gz$", fname):
            fd = gzip.open(fname, mode="r")
        else:
            fd = open(fname, mode="r")

        fname_duration = 0.0
        n_utts = 0
        for line in fd:
            js = json.loads(line)
            fname_duration += js["duration"]
            n_utts += 1

        print(
            f"Duration: {fname_duration/3600:7.2f} hours "
            f"(eq. {fname_duration:7.0f} seconds, {n_utts} utts): {fname}"
        )

        if fd != sys.stdin:
            fd.close()

        total_duration += fname_duration
        total_n_utts += n_utts

    print(
        f"Total duration: {total_duration/3600:7.2f} hours "
        f"(eq. {total_duration:7.0f} seconds)"
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
