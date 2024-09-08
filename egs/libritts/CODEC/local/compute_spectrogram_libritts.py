#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao,)
#              2024       The Chinese Univ. of HK  (authors: Zengrui Jin)
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
This file computes fbank features of the VCTK dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/spectrogram.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from lhotse import CutSet, LilcomChunkyWriter, Spectrogram, SpectrogramConfig
from lhotse.audio import RecordingSet
from lhotse.recipes.utils import read_manifests_if_cached
from lhotse.supervision import SupervisionSet

from icefall.utils import get_executor

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
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="""Sampling rate of the audio for computing fbank, the default value for LibriTTS is 24000, audio files will be resampled if a different sample rate is provided""",
    )

    return parser.parse_args()


def compute_spectrogram_libritts(
    dataset: Optional[str] = None, sampling_rate: int = 24000
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/spectrogram")
    num_jobs = min(32, os.cpu_count())

    frame_length = 1024 / sampling_rate  # (in second)
    frame_shift = 256 / sampling_rate  # (in second)
    use_fft_mag = True

    prefix = "libritts"
    suffix = "jsonl.gz"
    if dataset is None:
        dataset_parts = (
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    config = SpectrogramConfig(
        sampling_rate=sampling_rate,
        frame_length=frame_length,
        frame_shift=frame_shift,
        use_fft_mag=use_fft_mag,
    )
    extractor = Spectrogram(config)

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                return
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if sampling_rate != 24000:
                logging.info(f"Resampling audio to {sampling_rate}")
                cut_set = cut_set.resample(sampling_rate)

            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_spectrogram_libritts()
