#!/usr/bin/env python3
# Johns Hopkins University  (authors: Amir Hussein)
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
This file computes fbank features of the MGB2 dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path
import argparse

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

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

    return parser.parse_args()


def compute_fbank_gpu(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = os.cpu_count()
    num_mel_bins = 80
    sampling_rate = 16000
    sr = 16000

    dataset_parts = (
        "train",
        "test1",
        "dev",
    )
    manifests = read_manifests_if_cached(
        prefix="iwslt-ta", dataset_parts=dataset_parts, output_dir=src_dir
    )
    assert manifests is not None

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    for partition, m in manifests.items():
        if (output_dir / f"cuts_{partition}.jsonl.gz").is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        logging.info("About to split cuts into smaller chunks.")
        if sr != None:
            logging.info(f"Resampling to {sr}")
            cut_set = cut_set.resample(sr)

        cut_set = cut_set.trim_to_supervisions(
                    keep_overlapping=False, 
                    keep_all_channels=False)
        cut_set = cut_set.filter(lambda c: c.duration >= .2 and c.duration <= 30)
        if "train" in partition:
            cut_set = (
                cut_set
                + cut_set.perturb_speed(0.9)
                + cut_set.perturb_speed(1.1)
            )
            cut_set = cut_set.to_eager()
            chunk_size = len(cut_set) // args.num_splits
            cut_sets = cut_set.split_lazy(
                output_dir=src_dir / f"cuts_train_raw_split{args.num_splits}",
                chunk_size=chunk_size,)
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
                    batch_duration=1000,
                    num_workers=8,
                    storage_type=LilcomChunkyWriter,
                    overwrite=True,
                )
                cs.to_file(cuts_train_idx_path)
        else:
            logging.info(f"Processing {partition}")
            cut_set = cut_set.compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{partition}",
                batch_duration=1000,
                num_workers=10,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )
            cut_set.to_file(output_dir / f"cuts_{partition}.jsonl.gz")

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()

    compute_fbank_gpu(args)
