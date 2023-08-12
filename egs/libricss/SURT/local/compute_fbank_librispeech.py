#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import logging
from pathlib import Path

import torch
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


def compute_fbank_librispeech():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_mel_bins = 80

    dataset_parts = (
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    )
    prefix = "librispeech"
    suffix = "jsonl.gz"
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

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=16000),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    for partition, m in manifests.items():
        cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            manifest_path=f"{src_dir}/{cuts_filename}",
            batch_duration=4000,
            num_workers=2,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_fbank_librispeech()
