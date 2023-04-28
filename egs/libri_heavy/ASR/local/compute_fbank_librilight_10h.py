#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang)
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
This file computes fbank features of the LibriLight dataset.
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
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, KaldifeatFbank, KaldifeatFbankConfig, load_manifest
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
        "--dataset",
        type=str,
        default=None,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of dataloading workers used for reading the audio.",
    )
    
    return parser.parse_args()


def compute_fbank_librilight_10h(
    dataset: Optional[str] = None,
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(5, os.cpu_count())
    num_mel_bins = 80

    if dataset is None:
        dataset_parts = (
            "clean",
            "other",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for part in dataset_parts:
            output_cuts_path = output_dir / f"librilight_finetuning_{part}.jsonl.gz"
            if output_cuts_path.exists():
                logging.info(f"{output_cuts_path} exists - skipping")
                continue

            input_supervision_set = src_dir / f"librilight_finetuning_supervisions_{part}.jsonl.gz"
            assert input_supervision_set.exists(), f"{input_supervision_set} does not exist!"
            logging.info(f"Loading supervision set: {input_supervision_set}")
            
            input_recording_set = src_dir / f"librilight_finetuning_recordings_{part}.jsonl.gz"
            assert input_recording_set.exists(), f"{input_recording_set} does not exist!"
            logging.info(f"Loading recording set: {input_recording_set}")
            
            cut_set = CutSet.from_manifests(
                recordings=load_manifest(input_recording_set),
                supervisions=load_manifest(input_supervision_set)
            )

            logging.info("Computing features")

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/librilight_finetuning_feats_{part}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )

            logging.info(f"Saving to {output_cuts_path}")
            cut_set.to_file(output_cuts_path)
            
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))

    compute_fbank_librilight_10h(dataset=args.dataset)