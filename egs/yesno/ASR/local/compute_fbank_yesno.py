#!/usr/bin/env python3

"""
This file computes fbank features of the yesno dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or it wastes a
# lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_yesno():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")

    # This dataset is rather small, so we use only one job
    num_jobs = min(1, os.cpu_count())
    num_mel_bins = 23

    dataset_parts = (
        "train",
        "test",
    )
    prefix = "yesno"
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

    extractor = Fbank(FbankConfig(sampling_rate=8000, num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_file = output_dir / f"{prefix}_cuts_{partition}.{suffix}"
            if cuts_file.is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if "train" in partition:
                cut_set = (
                    cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
                )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 1,  # use one job
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(cuts_file)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_yesno()
