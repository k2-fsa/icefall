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
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomChunkyWriter,
)
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
        help="The number of splits of the medium and large subset.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Process pieces starting from this number (inclusive). Only used in medium and large subset",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="Stop processing pieces until this number (exclusive). Only used in medium and large subset",
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
        dataset_parts = ("small",)
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


def compute_fbank_libriheavy_splits(args):
    num_splits = args.num_splits
    dataset = args.dataset
    output_dir = f"data/fbank/libriheavy_{dataset}_split"
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

    prefix = "librilight"

    num_digits = 8  # num_digits is fixed by lhotse split-lazy
    for i in range(start, stop):
        idx = f"{i + 1}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"{prefix}_cuts_{dataset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = output_dir / f"{prefix}_cuts_{dataset}_raw.{idx}.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")
        if (output_dir / f"{prefix}_feats_{dataset}_{idx}.lca").exists():
            logging.info(f"Removing {output_dir}/{prefix}_feats_{dataset}_{idx}.lca")
            os.remove(output_dir / f"{prefix}_feats_{dataset}_{idx}.lca")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{dataset}_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    if args.dataset == "small":
        compute_fbank_libriheavy(
            bpe_model=args.bpe_model,
            dataset=args.dataset,
        )
    else:
        compute_fbank_libriheavy_splits(args)
