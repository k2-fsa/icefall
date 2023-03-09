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
This file computes fbank features of the synthetically mixed LibriSpeech
train and dev sets.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import logging
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import LilcomChunkyWriter
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


def compute_fbank_librimix():
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
    manifests = read_manifests_if_cached(
        dataset_parts=["train_norvb_v1", "dev_norvb_v1"],
        types=["cuts"],
        output_dir=src_dir,
        prefix="libri-mix",
        suffix="jsonl.gz",
        lazy=True,
    )

    train_cuts = manifests["train_norvb_v1"]["cuts"]
    dev_cuts = manifests["dev_norvb_v1"]["cuts"]
    # train_2spk_cuts = manifests["train_2spk_norvb"]["cuts"]

    logging.info("Extracting fbank features for training cuts")
    _ = train_cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=output_dir / "librimix_feats_train_norvb_v1",
        manifest_path=src_dir / "cuts_train_norvb_v1.jsonl.gz",
        batch_duration=5000,
        num_workers=4,
        storage_type=LilcomChunkyWriter,
        overwrite=True,
    )

    logging.info("Extracting fbank features for dev cuts")
    _ = dev_cuts.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=output_dir / "librimix_feats_dev_norvb_v1",
        manifest_path=src_dir / "cuts_dev_norvb_v1.jsonl.gz",
        batch_duration=5000,
        num_workers=4,
        storage_type=LilcomChunkyWriter,
        overwrite=True,
    )

    # logging.info("Extracting fbank features for 2-spk train cuts")
    # _ = train_2spk_cuts.compute_and_store_features_batch(
    #     extractor=extractor,
    #     storage_path=output_dir / "librimix_feats_train_2spk_norvb",
    #     manifest_path=src_dir / "cuts_train_2spk_norvb.jsonl.gz",
    #     batch_duration=5000,
    #     num_workers=4,
    #     storage_type=LilcomChunkyWriter,
    #     overwrite=True,
    # )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_fbank_librimix()
