#!/usr/bin/env python3

"""
This file computes fbank features of the librispeech dataset.
Its looks for manifests in the directory data/manifests
and generated fbank features are saved in data/fbank.
"""

import os
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, LilcomHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor


def compute_fbank_librispeech():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    )
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts, output_dir=src_dir
    )
    assert manifests is not None

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            if (output_dir / f"cuts_{partition}.json.gz").is_file():
                print(f"{partition} already exists - skipping.")
                continue
            print("Processing", partition)
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if "train" in partition:
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomHdf5Writer,
            )
            cut_set.to_json(output_dir / f"cuts_{partition}.json.gz")


if __name__ == "__main__":
    compute_fbank_librispeech()
