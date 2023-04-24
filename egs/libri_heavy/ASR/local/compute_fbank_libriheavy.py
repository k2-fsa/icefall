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

    return parser.parse_args()


def compute_fbank_libriheavy(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "small",
            "medium",
            "large",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for part in dataset_parts:
            output_cuts_path = output_dir / f"librilight_cuts_{part}.jsonl.gz"
            if output_cuts_path.exists():
                logging.info(f"{output_cuts_path} exists - skipping")
                continue

            input_cuts_path = src_dir / f"librilight_cuts_{part}.jsonl.gz"
            assert input_cuts_path.exists(), f"{input_cuts_path} does not exist!"
            logging.info(f"Loading {input_cuts_path}")
            cut_set = CutSet.from_file(input_cuts_path)

            logging.info("Computing features")

            if bpe_model:
                cut_set = filter_cuts(cut_set, sp)

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/libriheavy_feats_{part}",
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
    compute_fbank_libriheavy(
        bpe_model=args.bpe_model,
        dataset=args.dataset,
    )
