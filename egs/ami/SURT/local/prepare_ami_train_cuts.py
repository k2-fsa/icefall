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
This file creates AMI train segments.
"""
import logging
import math
from pathlib import Path

import torch
import torch.multiprocessing
from lhotse import LilcomChunkyWriter, load_manifest_lazy
from lhotse.cut import Cut, CutSet
from lhotse.utils import EPSILON, add_durations
from tqdm import tqdm


def cut_into_windows(cuts: CutSet, duration: float):
    """
    This function takes a CutSet and cuts each cut into windows of roughly
    `duration` seconds. By roughly, we mean that we try to adjust for the last supervision
    that exceeds the duration, or is shorter than the duration.
    """
    res = []
    with tqdm() as pbar:
        for cut in cuts:
            pbar.update(1)
            sups = cut.index_supervisions()[cut.id]
            sr = cut.sampling_rate
            start = 0.0
            end = duration
            num_tries = 0
            while start < cut.duration and num_tries < 2:
                # Find the supervision that are cut by the window endpoint
                hitlist = [iv for iv in sups.at(end) if iv.begin < end]
                # If there are no supervisions, we are done
                if not hitlist:
                    res.append(
                        cut.truncate(
                            offset=start,
                            duration=add_durations(end, -start, sampling_rate=sr),
                            keep_excessive_supervisions=False,
                        )
                    )
                    # Update the start and end for the next window
                    start = end
                    end = add_durations(end, duration, sampling_rate=sr)
                else:
                    # find ratio of durations cut by the window endpoint
                    ratios = [
                        add_durations(end, -iv.end, sampling_rate=sr) / iv.length()
                        for iv in hitlist
                    ]
                    # we retain the supervisions that have >50% of their duration
                    # in the window, and discard the others
                    retained = []
                    discarded = []
                    for iv, ratio in zip(hitlist, ratios):
                        if ratio > 0.5:
                            retained.append(iv)
                        else:
                            discarded.append(iv)
                    cur_end = max(iv.end for iv in retained) if retained else end
                    res.append(
                        cut.truncate(
                            offset=start,
                            duration=add_durations(cur_end, -start, sampling_rate=sr),
                            keep_excessive_supervisions=False,
                        )
                    )
                    # For the next window, we start at the earliest discarded supervision
                    next_start = min(iv.begin for iv in discarded) if discarded else end
                    next_end = add_durations(next_start, duration, sampling_rate=sr)
                    # It may happen that next_start is the same as start, in which case
                    # we will advance the window anyway
                    if next_start == start:
                        logging.warning(
                            f"Next start is the same as start: {next_start} == {start} for cut {cut.id}"
                        )
                        start = end + EPSILON
                        end = add_durations(start, duration, sampling_rate=sr)
                        num_tries += 1
                    else:
                        start = next_start
                        end = next_end
    return CutSet.from_cuts(res)


def prepare_train_cuts():
    src_dir = Path("data/manifests")

    logging.info("Loading the manifests")
    train_cuts_ihm = load_manifest_lazy(
        src_dir / "cuts_ami-ihm-mix_train.jsonl.gz"
    ).map(lambda c: c.with_id(f"{c.id}_ihm-mix"))
    train_cuts_sdm = load_manifest_lazy(src_dir / "cuts_ami-sdm_train.jsonl.gz").map(
        lambda c: c.with_id(f"{c.id}_sdm")
    )
    train_cuts_mdm = load_manifest_lazy(
        src_dir / "cuts_ami-mdm8-bf_train.jsonl.gz"
    ).map(lambda c: c.with_id(f"{c.id}_mdm8-bf"))

    # Combine all cuts into one CutSet
    train_cuts = train_cuts_ihm + train_cuts_sdm + train_cuts_mdm

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
    train_all.to_file(src_dir / "cuts_train_ami.jsonl.gz")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    prepare_train_cuts()
