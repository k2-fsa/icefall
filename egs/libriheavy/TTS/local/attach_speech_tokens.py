#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Yifan Yang)
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
import gzip
import json
import logging
import os
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        default="small",
    )

    return parser.parse_args()


def attach_speech_tokens(args):
    assert args.subset in ("small", "medium", "large"), f"{args.subset}"

    src_dir = Path("data/manifests")
    output_dir = "data/tokens"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    prefix = "libriheavy"

    cuts_path = output_dir / f"{prefix}_cuts_{args.subset}.jsonl.gz"
    if cuts_path.is_file():
        logging.info(f"{cuts_path} exists - skipping")
        return

    manifests_path = src_dir / f"{prefix}_cuts_{args.subset}.jsonl.gz"
    assert manifests_path.is_file(), f"{manifests_path} does not exist!"

    tokens_path = output_dir / f"{prefix}_{args.subset}.jsonl.gz"
    assert tokens_path.is_file(), f"{tokens_path} does not exist!"

    id2tokens = {}
    with gzip.open(tokens_path, "r") as fin:
        for line in fin:
            line = json.loads(line)
            id2tokens[line["key"]] = " ".join(map(str, line["code"]))

    with gzip.open(manifests_path, "r") as fin, gzip.open(cuts_path, "w") as fout:
        for cut in tqdm(fin, desc="Processing"):
            cut = json.loads(cut)
            if cut["id"] in id2tokens:
                cut["custom"] = {"tokens": id2tokens[cut["id"]]}
                fout.write((json.dumps(cut) + "\n").encode())


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    attach_speech_tokens(args)
