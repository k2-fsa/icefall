#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
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
This file computes fbank features of the WenetSpeech dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
import re
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomHdf5Writer,
    SupervisionSegment,
)
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--context-window",
        type=float,
        default=0.0,
        help="Training cut duration in seconds. "
        "Use 0 to train on supervision segments without acoustic context, "
        "with variable cut lengths; number larger than zero will create "
        "multi-supervisions cuts with actual acoustic context. ",
    )
    parser.add_argument(
        "--context-direction",
        type=str,
        default="center",
        help="If context-window is 0, does nothing. "
        "If it's larger than 0, determines in which direction "
        "(relative to the supervision) to seek for extra acoustic context. "
        "Available values: (left|right|center|random).",
    )
    parser.add_argument(
        "--precomputed-features",
        type=str2bool,
        default=False,
        help="Should we pre-compute features and store them on disk or not. "
        "It is recommended to disable it for L and XL splits as the "
        "pre-computation might currently consume excessive memory and time "
        "-- use on-the-fly feature extraction in the training script instead.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloading workers used for reading the audio.",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=600.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )
    return parser


# Similar text filtering and normalization procedure as in:
# https://github.com/SpeechColab/WenetSpeech/blob/main/toolkits/kaldi/wenetspeech_data_prep.sh


def normalize_text(
    utt: str,
    punct_pattern=re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


def has_no_oov(
    sup: SupervisionSegment,
    oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>"),
) -> bool:
    return oov_pattern.search(sup.text) is None


def get_context_suffix(args):
    if args.context_window is None or args.context_window <= 0.0:
        ctx_suffix = ""
    else:
        ctx_suffix = f"_{args.context_direction}{args.context_window}"
    return ctx_suffix


def compute_fbank_wenetspeech(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    dataset_parts = (
        "L",
        "M",
        "S",
        "DEV",
        "TEST_NET",
        "TEST_MEETING",
    )
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix="wenetspeech",
        suffix="jsonl.gz",
    )
    assert manifests is not None

    if torch.cuda.is_available():
        extractor = KaldifeatFbank(
            KaldifeatFbankConfig(device="cuda"),
        )
    else:
        extractor = KaldifeatFbank(
            KaldifeatFbankConfig(device="cpu"),
        )
    ctx_suffix = get_context_suffix(args)

    for partition, m in manifests.items():
        raw_cuts_path = output_dir / f"cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(
                f"{partition} already exists - skipping feature extraction."
            )
        else:
            # Note this step makes the recipe different than LibriSpeech:
            # We must filter out some utterances and remove punctuation
            # to be consistent with Kaldi.
            logging.info("Filtering OOV utterances from supervisions")
            m["supervisions"] = m["supervisions"].filter(has_no_oov)
            logging.info(f"Normalizing text in {partition}")
            for sup in m["supervisions"]:
                sup.text = normalize_text(sup.text)

            # Create long-recording cut manifests.
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            # Run data augmentation that needs to be done in the
            # time domain.
            if partition not in ["DEV", "TEST"]:
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cut_set.to_file(raw_cuts_path)

        cuts_path = output_dir / f"cuts_{partition}{ctx_suffix}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(
                f"{partition} already exists - skipping cutting into "
                f"sub-segments."
            )
        else:
            try:
                # If we skipped initializing `cut_set` because it exists
                # on disk, we'll load it. This helps us avoid re-computing
                # the features for different variants of context windows.
                cut_set
            except NameError:
                logging.info(f"Reading {partition} raw cuts from disk.")
                cut_set = CutSet.from_file(raw_cuts_path)
            # Note this step makes the recipe different than LibriSpeech:
            # Since recordings are long, the initial CutSet has very long
            # cuts with a plenty of supervisions. We cut these into smaller
            # chunks centered around each supervision, possibly adding
            # acoustic context.
            logging.info(
                f"About to split {partition} raw cuts into smaller chunks."
            )
            cut_set = cut_set.trim_to_supervisions(
                keep_overlapping=False,
                min_duration=None
                if args.context_window <= 0.0
                else args.context_window,
                context_direction=args.context_direction,
            )

            if args.precomputed_features:
                # Extract the features after cutting large recordings into
                # smaller cuts.
                # Note:
                # we support very efficient "chunked" feature reads with
                # the argument `storage_type=ChunkedLilcomHdf5Writer`,
                # but we don't support efficient data augmentation and
                # feature computation for long recordings yet.
                # Therefore, we sacrifice some storage for the ability to
                # precompute features on shorter chunks,
                # without memory blow-ups.
                if torch.cuda.is_available():
                    logging.info("GPU detected, do the CUDA extraction.")
                    cut_set = cut_set.compute_and_store_features_batch(
                        extractor=extractor,
                        storage_path=f"{output_dir}/feats_{partition}",
                        num_workers=args.num_workers,
                        batch_duration=args.batch_duration,
                        storage_type=LilcomHdf5Writer,
                    )
            cut_set.to_file(cuts_path)

            # Remove cut_set so the next iteration can correctly infer
            # whether it needs to load the raw cuts from disk or not.
            del cut_set

    # In case the user insists on CPU extraction
    if not torch.cuda.is_available():
        with get_executor() as ex:  # Initialize the executor only once.
            for partition, m in manifests.items():
                cuts_path = (
                    output_dir / f"cuts_{partition}{ctx_suffix}.jsonl.gz"
                )
                cut_set = CutSet.from_file(cuts_path)
                if args.precomputed_features:
                    # Extract the features after cutting large recordings into
                    # smaller cuts.
                    # Note:
                    # we support very efficient "chunked" feature reads with
                    # the argument `storage_type=ChunkedLilcomHdf5Writer`,
                    # but we don't support efficient data augmentation and
                    # feature computation for long recordings yet.
                    # Therefore, we sacrifice some storage for the ability to
                    # precompute features on shorter chunks,
                    # without memory blow-ups.
                    logging.info(
                        "GPU not detected, we recommend you skip the "
                        "extraction and do on-the-fly extraction "
                        "while training."
                    )
                    cut_set = cut_set.compute_and_store_features(
                        extractor=extractor,
                        storage_path=f"{output_dir}/feats_{partition}",
                        # when an executor is specified, make more partitions
                        num_jobs=min(15, os.cpu_count()) if ex is None else 80,
                        executor=ex,
                        storage_type=LilcomHdf5Writer,
                    )


def main():
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()

    compute_fbank_wenetspeech(args)


if __name__ == "__main__":
    main()
