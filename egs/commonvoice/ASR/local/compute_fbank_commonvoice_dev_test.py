#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Yifan Yang)
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
This file computes fbank features of the CommonVoice dataset.
It looks for manifests in the directory data/${lang}/manifests.

The generated fbank features are saved in data/${lang}/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from filter_cuts import filter_cuts
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig, LilcomChunkyWriter

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        type=str,
        help="""Language of Common Voice""",
    )

    return parser.parse_args()


def compute_fbank_commonvoice_dev_test(language: str):
    src_dir = Path(f"data/{language}/manifests")
    output_dir = Path(f"data/{language}/fbank")
    num_workers = 16
    batch_duration = 200

    subsets = ("dev", "test")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    for partition in subsets:
        cuts_path = output_dir / f"cv-{language}_cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue

        raw_cuts_path = output_dir / f"cv-{language}_cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Splitting cuts into smaller chunks")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/cv-{language}_feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_commonvoice_dev_test(language=args.language)
