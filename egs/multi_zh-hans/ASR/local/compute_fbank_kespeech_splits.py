#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
# Copyright    2023  Xiaomi Corp.             (Zengrui Jin)
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
    WhisperFbank,
    WhisperFbankConfig,
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


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--training-subset",
        type=str,
        default="train_phase1",
        choices=["train_phase1", "train_phase2"],
        help="The training subset for computing fbank feature.",
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
        help="The number of splits of the given subset",
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

    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bins for Fbank""",
    )

    parser.add_argument(
        "--whisper-fbank",
        type=str2bool,
        default=False,
        help="Use WhisperFbank instead of Fbank. Default: False.",
    )
    return parser


def compute_fbank_kespeech_splits(args):
    subset = args.training_subset
    subset = str(subset)
    num_splits = args.num_splits
    output_dir = f"data/fbank/kespeech/{subset}_split_{num_splits}"
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
    if args.whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=args.num_mel_bins, device="cuda")
        )
    else:
        extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    set_audio_duration_mismatch_tolerance(0.01)  # 10ms tolerance
    set_caching_enabled(False)
    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {i+1}/{num_splits}")

        cuts_path = output_dir / f"kespeech-asr_cuts_{subset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = output_dir / f"kespeech-asr_cuts_{subset}_raw.{idx}.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Splitting cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/feats_{subset}_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)


def main():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    log_filename = "log-compute_fbank_kespeech_splits"
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    log_filename = f"{log_filename}-{date_time}"

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=logging.INFO,
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(formatter))
    logging.getLogger("").addHandler(console)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    compute_fbank_kespeech_splits(args)


if __name__ == "__main__":
    main()
