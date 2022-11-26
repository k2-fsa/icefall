#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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
This file computes fbank features of the SPGISpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import argparse
import logging
from pathlib import Path

import torch
from lhotse import LilcomChunkyWriter, load_manifest_lazy
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
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
        "--num-splits",
        type=int,
        default=20,
        help="Number of splits for the train set.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the train set split.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop index of the train set split.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, only compute features for the dev and val set.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="If set, only compute features for the train set.",
    )

    return parser.parse_args()


def compute_fbank_spgispeech(args):
    assert args.train or args.test, "Either train or test must be set."

    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    if args.train:
        logging.info("Processing train")
        cut_set = load_manifest_lazy(src_dir / "cuts_train_raw.jsonl.gz")
        chunk_size = len(cut_set) // args.num_splits
        cut_sets = cut_set.split_lazy(
            output_dir=src_dir / f"cuts_train_raw_split{args.num_splits}",
            chunk_size=chunk_size,
        )
        start = args.start
        stop = min(args.stop, args.num_splits) if args.stop > 0 else args.num_splits
        num_digits = len(str(args.num_splits))
        for i in range(start, stop):
            idx = f"{i + 1}".zfill(num_digits)
            cuts_train_idx_path = src_dir / f"cuts_train_{idx}.jsonl.gz"
            logging.info(f"Processing train split {i}")
            cs = cut_sets[i].compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_train_{idx}",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )
            cs.to_file(cuts_train_idx_path)

    if args.test:
        for partition in ["dev", "val"]:
            if (output_dir / f"cuts_{partition}.jsonl.gz").is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = load_manifest_lazy(src_dir / f"cuts_{partition}_raw.jsonl.gz")
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{partition}",
                manifest_path=src_dir / f"cuts_{partition}.jsonl.gz",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_spgispeech(args)
