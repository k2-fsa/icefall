#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
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

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of dataloading workers used for reading the audio.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        required=True,
        help="The number of splits of the XL subset",
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


def compute_fbank_gigaspeech_splits(args):
    num_splits = args.num_splits
    output_dir = "data/fbank/gigaspeech_XL_split"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    start = args.start
    stop = args.stop
    if stop < start:
        stop = num_splits

    stop = min(stop, num_splits)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    num_digits = 8  # num_digits is fixed by lhotse split-lazy
    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"gigaspeech_cuts_XL.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = output_dir / f"gigaspeech_cuts_XL_raw.{idx}.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")
        filename = output_dir / f"gigaspeech_feats_XL_{idx}.lca"
        if filename.exists():
            logging.info(f"Removing {filename}")
            os.remove(str(filename))

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/gigaspeech_feats_XL_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_gigaspeech_splits(args)


if __name__ == "__main__":
    main()
