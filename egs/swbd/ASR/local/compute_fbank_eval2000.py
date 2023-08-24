#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# Modified     2023  The Chinese University of Hong Kong (author: Zengrui Jin)
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
This file computes fbank features of the SwitchBoard dataset.
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
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    return parser.parse_args()


def compute_fbank_switchboard(
    dir_name: str,
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    src_dir = Path(f"data/manifests/{dir_name}")
    output_dir = Path(f"data/fbank/{dir_name}")
    num_jobs = min(1, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = ("all",)
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = dir_name
    suffix = "jsonl.gz"
    manifests = {
        "eval2000": "data/manifests/eval2000/eval2000_cuts_all_trimmed.jsonl.gz",
    }
    assert manifests is not None

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins, sampling_rate=16000))

    with get_executor() as ex:  # Initialize the executor only once.
        partition = "all"
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        print(cuts_filename)
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{prefix} already exists - skipping.")
            return
        logging.info(f"Processing {prefix}")
        cut_set = CutSet.from_file(manifests[prefix]).resample(16000)

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_switchboard(
        dir_name="eval2000",
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
    )
