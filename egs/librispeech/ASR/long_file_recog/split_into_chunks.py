#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang, Zengwei Yao)
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
This script splits long utterances into chunks with overlaps.
Each chunk (except the first and the last) is padded with extra left side and right side.
The chunk length is: left_side + chunk_size + right_side.
"""

import argparse
import logging
from pathlib import Path

from lhotse import CutSet, load_manifest


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-in-dir",
        type=Path,
        default=Path("data/librilight/manifests"),
        help="Path to directory of full utterances.",
    )

    parser.add_argument(
        "--manifest-out-dir",
        type=Path,
        default=Path("data/librilight/manifests_chunk"),
        help="Path to directory to save splitted chunks.",
    )

    parser.add_argument(
        "--chunk",
        type=float,
        default=300.0,
        help="""Duration (in seconds) of each chunk.""",
    )

    parser.add_argument(
        "--extra",
        type=float,
        default=2.0,
        help="""Extra duration (in seconds) at both sides.""",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(vars(args))

    manifest_out_dir = args.manifest_out_dir
    manifest_out_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["small", "medium", "large"]

    for subset in subsets:
        logging.info(f"Processing {subset} subset")

        manifest_out = manifest_out_dir / f"librilight_cuts_{subset}.jsonl.gz"
        if manifest_out.is_file():
            logging.info(f"{manifest_out} already exists - skipping.")
            continue

        manifest_in = args.manifest_in_dir / f"librilight_recordings_{subset}.jsonl.gz"
        recordings = load_manifest(manifest_in)

        cuts = CutSet.from_manifests(recordings=recordings)
        cuts = cuts.cut_into_windows(
            duration=args.chunk, hop=args.chunk - args.extra * 2
        )
        cuts = cuts.fill_supervisions()

        cuts.to_file(manifest_out)
        logging.info(f"Cuts saved to {manifest_out}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
