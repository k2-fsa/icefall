#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This file computes fbank features of the LibriSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="data/manifests",
    )

    parser.add_argument(
        "--fbank-dir",
        type=str,
        default="data/fbank_mls",
    )

    parser.add_argument(
        "--part",
        type=str,
        help="Which language to prepare, if all, prepare all languages",
        choices=["english", "dutch", "german", "spanish", "french", "italian", "polish", "portuguese", "all"]
    )

    return parser.parse_args()

def compute_fbank_mls(
    manifest_dir=str,
    fbank_dir=str,
    part=str,
):
    src_dir = Path("data/manifests")
    output_dir = Path(fbank_dir)
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if part == "all":
        dataset_parts = [
            "english",
            "dutch",
            "german",
            "spanish"
        ]
    else:
        dataset_parts = [part]
    splits = ["train", "test", "dev"]

    num_jobs = 15
    num_mel_bins = 80
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    for language in dataset_parts:
        for split in splits:
            recording_file = src_dir / f"mls-{language}_recordings_{split}.jsonl.gz"
            supervision_file = src_dir / f"mls-{language}_supervisions_{split}.jsonl.gz"
            recordings = CutSet.from_file(recording_file)
            supervisions = CutSet.from_file(supervision_file)

            cut_set = CutSet.from_manifests(
                recordings=recordings,
                supervisions=supervisions,
            )

            prefix = f"mls-{language}"
            with get_executor() as ex:
                cut_set = cut_set.compute_and_store_features(
                    extractor=extractor,
                    storage_path=f"{output_dir}/{prefix}_feats_{split}",
                    # when an executor is specified, make more partitions
                    num_jobs=num_jobs if ex is None else 80,
                    executor=ex,
                    storage_type=LilcomChunkyWriter,
                )

            cuts_filename = output_dir / f"mls-{language}_{split}.jsonl.gz"
            
            logging.info(f"Saving to {cuts_filename}")
            cut_set.to_file(cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_mls(
        manifest_dir=args.manifest_dir,
        fbank_dir=args.fbank_dir,
        part=args.part,
    )


