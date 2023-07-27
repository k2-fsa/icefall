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
This file creates ICSI train segments.
"""
import logging
from pathlib import Path

from lhotse import load_manifest_lazy
from prepare_ami_train_cuts import cut_into_windows


def prepare_train_cuts():
    src_dir = Path("data/manifests")

    logging.info("Loading the manifests")
    train_cuts_ihm = load_manifest_lazy(
        src_dir / "cuts_icsi-ihm-mix_train.jsonl.gz"
    ).map(lambda c: c.with_id(f"{c.id}_ihm-mix"))
    train_cuts_sdm = load_manifest_lazy(src_dir / "cuts_icsi-sdm_train.jsonl.gz").map(
        lambda c: c.with_id(f"{c.id}_sdm")
    )

    # Combine all cuts into one CutSet
    train_cuts = train_cuts_ihm + train_cuts_sdm

    train_cuts_1 = train_cuts.trim_to_supervision_groups(max_pause=0.5)
    train_cuts_2 = train_cuts.trim_to_supervision_groups(max_pause=0.0)

    # Combine the two segmentations
    train_all = train_cuts_1 + train_cuts_2

    # At this point, some of the cuts may be very long. We will cut them into windows of
    # roughly 30 seconds.
    logging.info("Cutting the segments into windows of 30 seconds")
    train_all_30 = cut_into_windows(train_all, duration=30.0)
    logging.info(f"Number of cuts after cutting into windows: {len(train_all_30)}")

    # Show statistics
    train_all.describe(full=True)

    # Save the cuts
    logging.info("Saving the cuts")
    train_all.to_file(src_dir / "cuts_train_icsi.jsonl.gz")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    prepare_train_cuts()
