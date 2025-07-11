#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao)
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
This file computes fbank features of the baker-zh dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from fbank import MatchaFbank, MatchaFbankConfig
from lhotse import CutSet, LilcomChunkyWriter, load_manifest
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet

from icefall.utils import get_executor


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )
    return parser


def compute_fbank_baker_zh(num_jobs: int):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    if num_jobs < 1:
        num_jobs = os.cpu_count()

    logging.info(f"num_jobs: {num_jobs}")
    logging.info(f"src_dir: {src_dir}")
    logging.info(f"output_dir: {output_dir}")
    config = MatchaFbankConfig(
        n_fft=1024,
        n_mels=80,
        sampling_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0,
        f_max=8000,
    )
    if not torch.cuda.is_available():
        config.device = "cpu"

    prefix = "baker_zh"
    suffix = "jsonl.gz"

    extractor = MatchaFbank(config)

    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"{prefix}_cuts.{suffix}"
        logging.info(f"Processing {cuts_filename}")
        cut_set = load_manifest(src_dir / cuts_filename).resample(22050)

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats",
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    # Torch's multithreaded behavior needs to be disabled or
    # it wastes a lot of CPU and slow things down.
    # Do this outside of main() in case it needs to take effect
    # even when we are not invoking the main (e.g. when spawning subprocesses).
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_parser().parse_args()
    compute_fbank_baker_zh(args.num_jobs)
