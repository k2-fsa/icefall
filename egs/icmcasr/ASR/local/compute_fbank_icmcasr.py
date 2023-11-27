#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file computes fbank features of the icmcasr dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
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

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def compute_fbank_icmcasr(num_mel_bins: int = 80, perturb_speed: bool = False):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    manifests_ihm = read_manifests_if_cached(
        dataset_parts=["train", "dev"],
        output_dir=src_dir,
        prefix="icmcasr-ihm",
        suffix="jsonl.gz",
    )
    manifests_sdm = read_manifests_if_cached(
        dataset_parts=["train", "dev"],
        output_dir=src_dir,
        prefix="icmcasr-sdm",
        suffix="jsonl.gz",
    )
    # For GSS we already have cuts so we read them directly.
    manifests_gss = read_manifests_if_cached(
        dataset_parts=["train", "dev"],
        output_dir=src_dir,
        prefix="icmcasr-gss",
        suffix="jsonl.gz",
    )

    sampling_rate = 16000

    extractor = KaldifeatFbank(
        KaldifeatFbankConfig(
            frame_opts=KaldifeatFrameOptions(sampling_rate=sampling_rate),
            mel_opts=KaldifeatMelOptions(num_bins=num_mel_bins),
            device="cuda",
        )
    )

    def _extract_feats(
        cuts: CutSet, storage_path: Path, manifest_path: Path, speed_perturb: bool
    ) -> None:
        # check if the features have already been computed
        if storage_path.exists() or storage_path.with_suffix(".lca").exists():
            logging.info(f"{storage_path} exists, skipping feature extraction")
            return
        if speed_perturb:
            logging.info(f"Doing speed perturb")
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
        _ = cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=storage_path,
            manifest_path=manifest_path,
            batch_duration=5000,
            num_workers=8,
            storage_type=LilcomChunkyWriter,
        )

    logging.info(
        "Preparing training cuts: IHM + reverberated IHM + SDM + GSS (optional)"
    )

    logging.info("Processing train split IHM")
    cuts_ihm = (
        CutSet.from_manifests(**manifests_ihm["train"])
        .trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
        .modify_ids(lambda x: x + "-ihm")
    )
    _extract_feats(
        cuts_ihm,
        output_dir / "feats_train_ihm",
        src_dir / "cuts_train_ihm.jsonl.gz",
        perturb_speed,
    )

    logging.info("Processing train split IHM + reverberated IHM")
    cuts_ihm_rvb = cuts_ihm.reverb_rir()
    _extract_feats(
        cuts_ihm_rvb,
        output_dir / "feats_train_ihm_rvb",
        src_dir / "cuts_train_ihm_rvb.jsonl.gz",
        perturb_speed,
    )

    logging.info("Processing train split SDM")
    cuts_sdm = (
        CutSet.from_manifests(**manifests_sdm["train"])
        .trim_to_supervisions(keep_overlapping=False)
        .modify_ids(lambda x: x + "-sdm")
    )
    _extract_feats(
        cuts_sdm,
        output_dir / "feats_train_sdm",
        src_dir / "cuts_train_sdm.jsonl.gz",
        perturb_speed,
    )

    logging.info("Processing train split GSS")
    cuts_gss = (
        CutSet.from_manifests(**manifests_gss["train"])
        .trim_to_supervisions(keep_overlapping=False)
        .modify_ids(lambda x: x + "-gss")
    )
    _extract_feats(
        cuts_gss,
        output_dir / "feats_train_gss",
        src_dir / "cuts_train_gss.jsonl.gz",
        perturb_speed,
    )

    logging.info("Preparing test cuts: IHM, SDM, GSS (optional)")
    for split in ["dev"]:
        logging.info(f"Processing {split} IHM")
        cuts_ihm = (
            CutSet.from_manifests(**manifests_ihm[split])
            .trim_to_supervisions(keep_overlapping=False, keep_all_channels=False)
            .compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{split}_ihm",
                manifest_path=src_dir / f"cuts_{split}_ihm.jsonl.gz",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
            )
        )
        logging.info(f"Processing {split} SDM")
        cuts_sdm = (
            CutSet.from_manifests(**manifests_sdm[split])
            .trim_to_supervisions(keep_overlapping=False)
            .compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{split}_sdm",
                manifest_path=src_dir / f"cuts_{split}_sdm.jsonl.gz",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
            )
        )
        logging.info(f"Processing {split} GSS")
        cuts_gss = (
            CutSet.from_manifests(**manifests_gss[split])
            .trim_to_supervisions(keep_overlapping=False)
            .compute_and_store_features_batch(
                extractor=extractor,
                storage_path=output_dir / f"feats_{split}_gss",
                manifest_path=src_dir / f"cuts_{split}_gss.jsonl.gz",
                batch_duration=500,
                num_workers=4,
                storage_type=LilcomChunkyWriter,
            )
        )



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="""The number of mel bins for Fbank""",
    )
    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="Enable 0.9 and 1.1 speed perturbation for data augmentation. Default: False.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    compute_fbank_icmcasr(
        num_mel_bins=args.num_mel_bins, perturb_speed=args.perturb_speed
    )
