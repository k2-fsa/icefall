#!/usr/bin/env python3

"""
This file computes fbank features of the musan dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of CPU and
# slow things down.  Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_musan():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        "music",
        "speech",
        "noise",
    )
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts, output_dir=src_dir
    )
    assert manifests is not None

    musan_cuts_path = output_dir / "cuts_musan.json.gz"

    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for Musan")

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        # create chunks of Musan with duration 5 - 10 seconds
        musan_cuts = (
            CutSet.from_manifests(
                recordings=combine(
                    part["recordings"] for part in manifests.values()
                )
            )
            .cut_into_windows(10.0)
            .filter(lambda c: c.duration > 5)
            .compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/feats_musan",
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer,
            )
        )
        musan_cuts.to_json(musan_cuts_path)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    compute_fbank_musan()
