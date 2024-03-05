#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
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

import argparse
import logging
from pathlib import Path

import lhotse
import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    fix_manifests,
    validate_recordings_and_supervisions,
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
        "--kaldi-dir",
        type=str,
        help="""The directory containing kaldi style manifest, namely wav.scp, text and segments.
        """,
    )

    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bank bins.
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/fbank",
        help="""The directory where the lhotse manifests and features to write to.
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""The name of dataset.
        """,
    )

    parser.add_argument(
        "--partition",
        type=str,
        help="""Could be something like train, valid, test and so on.
        """,
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    parser.add_argument(
        "--num-jobs", type=int, default=50, help="The num of jobs to extract feature."
    )

    return parser.parse_args()


def prepare_cuts(args):
    logging.info(f"Prepare cuts from {args.kaldi_dir}.")
    recordings, supervisions, _ = lhotse.load_kaldi_data_dir(args.kaldi_dir, 16000)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)
    cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
    return cuts


def compute_feature(args, cuts):
    extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins))
    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"{args.dataset}_cuts_{args.partition}.jsonl.gz"
        if (args.output_dir / cuts_filename).is_file():
            logging.info(f"{cuts_filename} already exists - skipping.")
            return
        logging.info(f"Processing {cuts_filename}")

        if "train" in args.partition:
            if args.perturb_speed:
                logging.info(f"Doing speed perturb")
                cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        cuts = cuts.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{args.output_dir}/{args.dataset}_feats_{args.partition}",
            # when an executor is specified, make more partitions
            num_jobs=args.num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        cuts.to_file(args.output_dir / cuts_filename)


def main(args):
    args.kaldi_dir = Path(args.kaldi_dir)
    args.output_dir = Path(args.output_dir)
    cuts = prepare_cuts(args)
    compute_feature(args, cuts)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    main(args)
