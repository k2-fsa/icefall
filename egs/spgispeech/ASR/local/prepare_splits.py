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
This file splits the training set into train and dev sets.
"""
import logging
from pathlib import Path

import torch
from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def split_spgispeech_train():
    src_dir = Path("data/manifests")

    manifests = read_manifests_if_cached(
        dataset_parts=["train", "val"],
        output_dir=src_dir,
        prefix="spgispeech",
        suffix="jsonl.gz",
        lazy=True,
    )
    assert manifests is not None

    train_dev_cuts = CutSet.from_manifests(
        recordings=manifests["train"]["recordings"],
        supervisions=manifests["train"]["supervisions"],
    )
    dev_cuts = train_dev_cuts.subset(first=4000)
    train_cuts = train_dev_cuts.filter(lambda c: c not in dev_cuts)

    # Add speed perturbation
    train_cuts = (
        train_cuts + train_cuts.perturb_speed(0.9) + train_cuts.perturb_speed(1.1)
    )

    # Write the manifests to disk.
    train_cuts.to_file(src_dir / "cuts_train_raw.jsonl.gz")
    dev_cuts.to_file(src_dir / "cuts_dev_raw.jsonl.gz")

    # Also write the val set to disk.
    val_cuts = CutSet.from_manifests(
        recordings=manifests["val"]["recordings"],
        supervisions=manifests["val"]["supervisions"],
    )
    val_cuts.to_file(src_dir / "cuts_val_raw.jsonl.gz")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    split_spgispeech_train()
