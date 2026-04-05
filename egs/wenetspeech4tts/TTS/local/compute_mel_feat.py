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
This file computes fbank features of the LJSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from fbank import MatchaFbank, MatchaFbankConfig
from lhotse import CutSet, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fbank"),
        help="Path to the tokenized files",
    )

    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="Basic",
        help="Space separated dataset parts",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="wenetspeech4tts",
        help="prefix of the manifest file",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )

    parser.add_argument(
        "--split",
        type=int,
        default=100,
        help="Split the cut_set into multiple parts",
    )

    parser.add_argument(
        "--resample-to-24kHz",
        default=True,
        help="Resample the audio to 24kHz",
    )

    parser.add_argument(
        "--extractor",
        type=str,
        choices=["bigvgan", "hifigan"],
        default="bigvgan",
        help="The type of extractor to use",
    )
    return parser


def compute_fbank(args):
    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    num_jobs = min(args.num_jobs, os.cpu_count())
    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip().split(" ")

    logging.info(f"num_jobs: {num_jobs}")
    logging.info(f"src_dir: {src_dir}")
    logging.info(f"output_dir: {output_dir}")
    logging.info(f"dataset_parts: {dataset_parts}")
    if args.extractor == "bigvgan":
        config = MatchaFbankConfig(
            n_fft=1024,
            n_mels=100,
            sampling_rate=24_000,
            hop_length=256,
            win_length=1024,
            f_min=0,
            f_max=None,
        )
    elif args.extractor == "hifigan":
        config = MatchaFbankConfig(
            n_fft=1024,
            n_mels=80,
            sampling_rate=22050,
            hop_length=256,
            win_length=1024,
            f_min=0,
            f_max=8000,
        )
    else:
        raise NotImplementedError(f"Extractor {args.extractor} is not implemented")

    extractor = MatchaFbank(config)

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )

    with get_executor() as ex:
        for partition, m in manifests.items():
            logging.info(
                f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
            )
            try:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )
            except Exception:
                cut_set = m["cuts"]

            if args.split > 1:
                cut_sets = cut_set.split(args.split)
            else:
                cut_sets = [cut_set]

            for idx, part in enumerate(cut_sets):
                if args.split > 1:
                    storage_path = f"{args.output_dir}/{args.prefix}_{args.extractor}_{partition}_{idx}"
                else:
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_{args.extractor}_{partition}"
                    )

                if args.resample_to_24kHz:
                    part = part.resample(24000)

                with torch.no_grad():
                    part = part.compute_and_store_features(
                        extractor=extractor,
                        storage_path=storage_path,
                        num_jobs=num_jobs if ex is None else 64,
                        executor=ex,
                        storage_type=LilcomChunkyWriter,
                    )

                if args.split > 1:
                    cuts_filename = (
                        f"{args.prefix}_cuts_{partition}.{idx}.{args.suffix}"
                    )
                else:
                    cuts_filename = f"{args.prefix}_cuts_{partition}.{args.suffix}"

                part.to_file(f"{args.output_dir}/{cuts_filename}")
                logging.info(f"Saved {cuts_filename}")


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
    compute_fbank(args)
