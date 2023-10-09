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
This file computes fbank features of the trimmed sub-segments which will be
used for simulating the training mixtures.

The generated fbank features are saved in data/fbank.
"""
import logging
import math
from pathlib import Path

import torch
import torch.multiprocessing
import torchaudio
from lhotse import CutSet, LilcomChunkyWriter, load_manifest
from lhotse.audio import set_ffmpeg_torchaudio_info_enabled
from lhotse.features.kaldifeat import (
    KaldifeatFbank,
    KaldifeatFbankConfig,
    KaldifeatFrameOptions,
    KaldifeatMelOptions,
)
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")
torchaudio.set_audio_backend("soundfile")
set_ffmpeg_torchaudio_info_enabled(False)


def compute_fbank_ihm():
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
    for data in ["ami", "icsi"]:
        manifests[data] = read_manifests_if_cached(
            dataset_parts=["train"],
            output_dir=src_dir,
            types=["recordings", "supervisions"],
            prefix=f"{data}-ihm",
            suffix="jsonl.gz",
        )

    logging.info("Computing features")
    for data in ["ami", "icsi"]:
        cs = CutSet.from_manifests(**manifests[data]["train"])
        cs = cs.trim_to_supervisions(keep_overlapping=False)
        cs = cs.normalize_loudness(target=-23.0, affix_id=False)
        cs = cs + cs.perturb_speed(0.9) + cs.perturb_speed(1.1)
        _ = cs.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=output_dir / f"{data}-ihm_train_feats",
            manifest_path=src_dir / f"{data}-ihm_cuts_train.jsonl.gz",
            batch_duration=5000,
            num_workers=4,
            storage_type=LilcomChunkyWriter,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_ihm()
