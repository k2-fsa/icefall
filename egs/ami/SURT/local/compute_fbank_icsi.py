#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University        (authors: Desh Raj)
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
This file computes fbank features of the ICSI dataset.
We compute features for full recordings (i.e., without trimming to supervisions).
This way we can create arbitrary segmentations later.

The generated fbank features are saved in data/fbank.
"""
import logging
import math
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import CutSet, LilcomChunkyWriter
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fbank_icsi():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    sampling_rate = 16000
    num_mel_bins = 80

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    logging.info("Reading manifests")
    manifests = {}
    for part in ["ihm-mix", "sdm"]:
        manifests[part] = read_manifests_if_cached(
            dataset_parts=["train"],
            output_dir=src_dir,
            prefix=f"icsi-{part}",
            suffix="jsonl.gz",
        )

    for part in ["ihm-mix", "sdm"]:
        for split in ["train"]:
            logging.info(f"Processing {part} {split}")
            cuts = CutSet.from_manifests(
                **manifests[part][split]
            ).compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"icsi-{part}_{split}_feats",
                manifest_path=src_dir / f"cuts_icsi-{part}_{split}.jsonl.gz",
                batch_duration=5000,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
                overwrite=True,
            )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_icsi()
