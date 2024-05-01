#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.             (Yifan Yang)
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
from datetime import datetime
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

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

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
        help="The number of splits of the train subset",
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

    return parser.parse_args()


def compute_fbank_peoples_speech_splits(args):
    subsets = ("dirty", "dirty_sa", "clean", "clean_sa")
    num_splits = args.num_splits
    output_dir = f"data/fbank/peoples_speech_train_split"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    num_digits = 8

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

    set_audio_duration_mismatch_tolerance(0.01)  # 10ms tolerance
    set_caching_enabled(False)

    for partition in subsets:
        for i in range(start, stop):
            idx = f"{i + 1}".zfill(num_digits)
            logging.info(f"Processing {partition}: {idx}")

            cuts_path = output_dir / f"peoples_speech_cuts_{partition}.{idx}.jsonl.gz"
            if cuts_path.is_file():
                logging.info(f"{cuts_path} exists - skipping")
                continue

            raw_cuts_path = (
                output_dir / f"peoples_speech_cuts_{partition}_raw.{idx}.jsonl.gz"
            )

            logging.info(f"Loading {raw_cuts_path}")
            cut_set = CutSet.from_file(raw_cuts_path)

            logging.info("Splitting cuts into smaller chunks.")
            cut_set = cut_set.trim_to_supervisions(
                keep_overlapping=False, min_duration=None
            )

            logging.info("Computing features")
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=f"{output_dir}/peoples_speech_feats_{partition}_{idx}",
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
    compute_fbank_peoples_speech_splits(args)


if __name__ == "__main__":
    main()
