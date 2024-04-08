#!/usr/bin/env python3
# Copyright    2023-2024  Xiaomi Corp.             (Yifan Yang,
#                                                   Zengrui Jin,)
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
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomChunkyWriter,
    set_audio_duration_mismatch_tolerance,
    set_caching_enabled,
)

from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "validated", "invalidated"],
        help="""Dataset parts to compute fbank. """,
    )

    parser.add_argument(
        "--language",
        type=str,
        help="""Language of Common Voice""",
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
        help="The number of splits of the subset",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (included).",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (excluded).",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    return parser.parse_args()


def compute_fbank_commonvoice_splits(args):
    subset = args.subset
    num_splits = args.num_splits
    language = args.language
    output_dir = f"data/{language}/fbank/cv-{language}_{subset}_split_{num_splits}"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    num_digits = len(str(num_splits))

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

    set_audio_duration_mismatch_tolerance(0.05)  # 50ms tolerance
    set_caching_enabled(False)
    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"cv-{language}_cuts_{subset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = output_dir / f"cv-{language}_cuts_{subset}_raw.{idx}.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Splitting cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        if args.perturb_speed:
            logging.info(f"Doing speed perturb")
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/cv-{language}_feats_{subset}_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_commonvoice_splits(args)


if __name__ == "__main__":
    main()
