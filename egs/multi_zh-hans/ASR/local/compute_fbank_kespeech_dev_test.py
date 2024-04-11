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
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomChunkyWriter,
    WhisperFbank,
    WhisperFbankConfig,
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


def compute_fbank_kespeech_dev_test(args):
    in_out_dir = Path("data/fbank/kespeech")
    # number of workers in dataloader
    num_workers = 42

    # number of seconds in a batch
    batch_duration = 600

    subsets = (
        "dev_phase1",
        "dev_phase2",
        "test",
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    if args.whisper_fbank:
        extractor = WhisperFbank(
            WhisperFbankConfig(num_filters=args.num_mel_bins, device=device)
        )
    else:
        extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    for partition in subsets:
        cuts_path = in_out_dir / f"kespeech-asr_cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = in_out_dir / f"kespeech-asr_cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Splitting cuts into smaller chunks")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info("Computing features")
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{in_out_dir}/feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    compute_fbank_kespeech_dev_test(args)


if __name__ == "__main__":
    main()
