#!/usr/bin/env python3
# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao,
#                                                       Zengrui Jin,)
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

import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    LilcomChunkyWriter,
    Spectrogram,
    SpectrogramConfig,
    load_manifest,
)
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSet

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_spectrogram_vctk():
    src_dir = Path("data/manifests")
    output_dir = Path("data/spectrogram")
    num_jobs = min(32, os.cpu_count())

    sampling_rate = 22050
    frame_length = 1024 / sampling_rate  # (in second)
    frame_shift = 256 / sampling_rate  # (in second)
    use_fft_mag = True

    prefix = "vctk"
    suffix = "jsonl.gz"
    partition = "all"

    recordings = load_manifest(
        src_dir / f"{prefix}_recordings_{partition}.jsonl.gz", RecordingSet
    ).resample(sampling_rate=sampling_rate)
    supervisions = load_manifest(
        src_dir / f"{prefix}_supervisions_{partition}.jsonl.gz", SupervisionSet
    )

    config = SpectrogramConfig(
        sampling_rate=sampling_rate,
        frame_length=frame_length,
        frame_shift=frame_shift,
        use_fft_mag=use_fft_mag,
    )
    extractor = Spectrogram(config)

    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{partition} already exists - skipping.")
            return
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=recordings, supervisions=supervisions
        )

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
    compute_spectrogram_vctk()
