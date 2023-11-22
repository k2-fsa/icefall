#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Xiaoyu Yang,
#                                                  Wei Kang)
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
This file computes fbank features of the Libriheavy dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomChunkyWriter,
)

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
        help="""The source directory that contains raw manifests.
        """,
        default="data/manifests",
    )

    parser.add_argument(
        "--fbank-dir",
        type=str,
        help="""Fbank output dir
        """,
        default="data/fbank",
    )

    parser.add_argument(
        "--subset",
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
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="Whether to use speed perturbation.",
    )

    parser.add_argument(
        "--use-splits",
        type=str2bool,
        default=False,
        help="Whether to compute fbank on splits.",
    )

    parser.add_argument(
        "--num-splits",
        type=int,
        help="""The number of splits of the medium and large subset.
        Only needed when --use-splits is true.""",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="""Process pieces starting from this number (inclusive).
        Only needed when --use-splits is true.""",
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=-1,
        help="""Stop processing pieces until this number (exclusive).
        Only needed when --use-splits is true.""",
    )

    return parser.parse_args()


def compute_fbank_libriheavy(args):
    src_dir = Path(args.manifest_dir)
    output_dir = Path(args.fbank_dir)
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80
    subset = args.subset

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        output_cuts_path = output_dir / f"libriheavy_cuts_{subset}.jsonl.gz"
        if output_cuts_path.exists():
            logging.info(f"{output_cuts_path} exists - skipping")
            return

        input_cuts_path = src_dir / f"libriheavy_cuts_{subset}.jsonl.gz"
        assert input_cuts_path.exists(), f"{input_cuts_path} does not exist!"
        logging.info(f"Loading {input_cuts_path}")
        cut_set = CutSet.from_file(input_cuts_path)

        logging.info("Computing features")

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/libriheavy_feats_{subset}",
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

        logging.info(f"Saving to {output_cuts_path}")
        cut_set.to_file(output_cuts_path)


def compute_fbank_libriheavy_splits(args):
    num_splits = args.num_splits
    subset = args.subset
    src_dir = f"{args.manifest_dir}/libriheavy_{subset}_split"
    src_dir = Path(src_dir)
    output_dir = f"{args.fbank_dir}/libriheavy_{subset}_split"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    num_digits = 8  # num_digits is fixed by lhotse split-lazy
    for i in range(start, stop):
        idx = f"{i + 1}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"libriheavy_cuts_{subset}.{idx}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = src_dir / f"libriheavy_cuts_{subset}.{idx}.jsonl.gz"
        if not raw_cuts_path.is_file():
            logging.info(f"{raw_cuts_path} does not exist - skipping it")
            continue

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")
        if (output_dir / f"libriheavy_feats_{subset}_{idx}.lca").exists():
            logging.info(f"Removing {output_dir}/libriheavy_feats_{subset}_{idx}.lca")
            os.remove(output_dir / f"libriheavy_feats_{subset}_{idx}.lca")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/libriheavy_feats_{subset}_{idx}",
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

    if args.use_splits:
        assert args.num_splits is not None, "Please provide num_splits"
        compute_fbank_libriheavy_splits(args)
    else:
        compute_fbank_libriheavy(args)
