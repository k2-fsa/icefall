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
This file computes fbank features of the LibriCSS dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""
import logging
from pathlib import Path

import pyloudnorm as pyln
import torch
import torch.multiprocessing
from lhotse import LilcomChunkyWriter, load_manifest_lazy
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def compute_fbank_libricss():
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
    cuts_ihm_mix = load_manifest_lazy(
        src_dir / "libricss-ihm-mix_segments_all.jsonl.gz"
    )
    cuts_sdm = load_manifest_lazy(src_dir / "libricss-sdm_segments_all.jsonl.gz")

    for name, cuts in [("ihm-mix", cuts_ihm_mix), ("sdm", cuts_sdm)]:
        dev_cuts = cuts.filter(lambda c: "session0" in c.id)
        test_cuts = cuts.filter(lambda c: "session0" not in c.id)

        # If SDM cuts, apply loudness normalization
        if name == "sdm":
            dev_cuts = dev_cuts.normalize_loudness(target=-23.0)
            test_cuts = test_cuts.normalize_loudness(target=-23.0)

        logging.info(f"Extracting fbank features for {name} dev cuts")
        _ = dev_cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / f"libricss-{name}_feats_dev",
            manifest_path=src_dir / f"cuts_dev_libricss-{name}.jsonl.gz",
            batch_duration=500,
            num_workers=2,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )

        logging.info(f"Extracting fbank features for {name} test cuts")
        _ = test_cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / f"libricss-{name}_feats_test",
            manifest_path=src_dir / f"cuts_test_libricss-{name}.jsonl.gz",
            batch_duration=2000,
            num_workers=4,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_libricss()
