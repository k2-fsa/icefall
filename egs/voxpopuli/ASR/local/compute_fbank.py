#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2023  Brno University of Technology  (authors: Karel Veselý)
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
This file computes fbank features of VoxPopuli dataset.

Usage example:

  python3 ./local/compute_fbank.py \
      --src-dir data/fbank --output-dir data/fbank \
      --num-jobs 100 --num-workers 25 \
      --prefix "voxpopuli-${task}-${lang}" \
      --dataset train \
      --trim-to-supervisions True \
      --speed-perturb True

It looks for raw CutSet in the directory data/fbank
located at: `{src_dir}/{prefix}_cuts_{dataset}_raw.jsonl.gz`.

The generated fbank features are saved in `data/fbank/{prefix}-{dataset}_feats`
and CutSet manifest stored in `data/fbank/{prefix}_cuts_{dataset}.jsonl.gz`.

Typically, the number of workers is smaller than number of jobs
(see --num-jobs 100 --num-workers 25 in the example).
And, the number of jobs should be at least the number of workers (it's checked).
"""

import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    is_caching_enabled,
    set_caching_enabled,
)

from icefall.utils import str2bool

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
        "--src-dir",
        type=str,
        help="""Folder with the input manifest files.""",
        default="data/manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="""Folder with the output manifests (cuts) and feature files.""",
        default="data/fbank",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        help="""Prefix of the manifest files.""",
        default="",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank (train,test,dev).""",
        default=None,
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        help="""Number of jobs (i.e. files with extracted features)""",
        default=50,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="""Number of parallel workers""",
        default=10,
    )
    parser.add_argument(
        "--speed-perturb",
        type=str2bool,
        default=False,
        help="""Enable speed perturbation for the set.""",
    )
    parser.add_argument(
        "--trim-to-supervisions",
        type=str2bool,
        default=False,
        help="""Apply `trim-to-supervision` to cut set.""",
    )

    return parser.parse_args()


def compute_fbank_features(args: argparse.Namespace):
    set_caching_enabled(True)  # lhotse

    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)
    num_jobs = args.num_jobs
    num_workers = min(args.num_workers, os.cpu_count())
    num_mel_bins = 80

    bpe_model = args.bpe_model
    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    prefix = args.prefix  # "ELEF_TRAIN"
    dataset = args.dataset
    suffix = "jsonl.gz"

    cuts_raw_filename = Path(f"{src_dir}/{prefix}_cuts_{dataset}_raw.{suffix}")
    cuts_raw = CutSet.from_file(cuts_raw_filename)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    cuts_filename = Path(f"{prefix}_cuts_{dataset}.{suffix}")
    if (output_dir / cuts_filename).is_file():
        logging.info(f"{output_dir/cuts_filename} already exists - skipping.")
        return

    logging.info(f"Processing {output_dir/cuts_filename}")
    cut_set = cuts_raw

    if bpe_model:
        cut_set = filter_cuts(cut_set, sp)

    if args.speed_perturb:
        cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

    if args.trim_to_supervisions:
        logging.info(f"About to `trim_to_supervisions()` {output_dir / cuts_filename}")
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    else:
        logging.info(
            "Not doing `trim_to_supervisions()`, "
            "to enable use --trim-to-supervision=True"
        )

    cut_set = cut_set.to_eager()  # disallow lazy evaluation (sorting requires it)
    cut_set = cut_set.sort_by_recording_id()  # enhances AudioCache hit rate

    # We typically use `num_jobs=100, num_workers=20`
    # - this is helpful for large databases
    # - both values are configurable externally
    assert num_jobs >= num_workers, (num_jobs, num_workers)
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=set_caching_enabled,
        initargs=(is_caching_enabled(),),
    )

    logging.info(
        f"executor {executor} : num_workers {num_workers}, num_jobs {num_jobs}"
    )

    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{output_dir / prefix}-{dataset}_feats",
        num_jobs=num_jobs,
        executor=executor,
        storage_type=LilcomChunkyWriter,
    )

    # correct small deviations of duration, caused by speed-perturbation
    for cut in cut_set:
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut.id)
        duration_difference = abs(cut.supervisions[0].duration - cut.duration)
        tolerance = 0.02  # 20ms
        if duration_difference == 0.0:
            pass
        elif duration_difference <= tolerance:
            logging.info(
                "small mismatch of the supervision duration "
                f"(Δt = {duration_difference*1000}ms), "
                f"correcting : cut.duration {cut.duration} -> "
                f"supervision {cut.supervisions[0].duration}"
            )
            cut.supervisions[0].duration = cut.duration
        else:
            logging.error(
                "mismatch of cut/supervision duration "
                f"(Δt = {duration_difference*1000}ms) : "
                f"cut.duration {cut.duration}, "
                f"supervision {cut.supervisions[0].duration}"
            )
            raise ValueError(
                "mismatch of cut/supervision duration "
                f"(Δt = {duration_difference*1000}ms)"
            )

    # store the cutset
    logging.info(f"storing CutSet to : `{output_dir / cuts_filename}`")
    cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    logging.info(vars(args))

    compute_fbank_features(args)
