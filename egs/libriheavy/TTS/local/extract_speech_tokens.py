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
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.utils import fastcopy
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        default="small",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="download/hubert_base_ls960.pt",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive).",
    )

    return parser.parse_args()


@torch.no_grad()
def extract_and_save_one_cuts(
    manifests_path,
    cuts_path,
):
    logging.info(f"Loading {manifests_path}")
    cut_set = CutSet.from_file(manifests_path)

    logging.info("Extracting tokens")
    cuts = []

    tokens = " ".join(map(str, tokens))

    cut_with_tokens = fastcopy(
        cut,
        custom={"tokens": tokens},
    )
    cuts.append(cut_with_tokens)

    cuts = CutSet(cuts)

    logging.info(f"Saving to {cuts_path}")
    cuts.to_file(cuts_path)


def extract_speech_tokens(args):
    assert args.subset in ("small", "medium", "large"), f"{args.subset}"

    output_dir = (
        f"data/tokens/{args.subset}_split" if args.subset != "small" else "data/tokens"
    )
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"device: {device}")

    prefix = "libriheavy"

    if args.subset == "small":
        cuts_path = output_dir / f"{prefix}_cuts_{args.subset}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            return

        manifests_path = output_dir / f"{prefix}_cuts_{args.subset}.jsonl.gz"
        if not manifests_path.is_file():
            logging.info(f"{manifests_path} does not exist - skipping it")
            return

        extract_and_save_one_cuts(
            manifests_path,
            cuts_path,
            model,
            apply_tokens,
            do_normalize,
            window_duration,
            shift_duration,
        )
    else:
        num_digits = 8  # num_digits is fixed by lhotse split-lazy
        start = args.start
        stop = args.stop
        assert stop > start, "stop must be larger than start!"

        for i in range(start, stop):
            idx = f"{i}".zfill(num_digits)
            logging.info(f"Processing {idx}/{stop - 1}")

            cuts_path = output_dir / f"{prefix}_cuts_{args.subset}.{idx}.jsonl.gz"
            if cuts_path.is_file():
                logging.info(f"{cuts_path} exists - skipping")
                continue

            manifests_path = (
                output_dir / f"{prefix}_cuts_{args.subset}.{idx}.jsonl.gz"
            )
            if not manifests_path.is_file():
                logging.info(f"{manifests_path} does not exist - skipping it")
                continue

            extract_and_save_one_cuts(
                manifests_path,
                cuts_path,
            )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    extract_speech_tokens(args)
