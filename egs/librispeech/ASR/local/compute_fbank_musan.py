#!/usr/bin/env python3

"""
This file computes fbank features of the musan dataset.
Its looks for manifests in the directory data/manifests
and generated fbank features are saved in data/fbank.
"""

import os
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor


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
        print(f"{musan_cuts_path} already exists - skipping")
        return

    print("Extracting features for Musan")

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
    compute_fbank_musan()
