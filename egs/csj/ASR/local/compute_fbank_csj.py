#!/usr/bin/env python3
# Copyright    2022  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
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


import argparse
import logging
import os
from itertools import islice
from pathlib import Path
from random import Random
from typing import List, Tuple

import torch

# fmt: off
from lhotse import (  # See the following for why LilcomChunkyWriter is preferred; https://github.com/k2-fsa/icefall/pull/404; https://github.com/lhotse-speech/lhotse/pull/527
    CutSet,
    Fbank,
    FbankConfig,
    LilcomChunkyWriter,
    RecordingSet,
    SupervisionSet,
)

# fmt: on

ARGPARSE_DESCRIPTION = """
This script follows the espnet method of splitting the remaining core+noncore
utterances into valid and train cutsets at an index which is by default 4000.

In other words, the core+noncore utterances are shuffled, where 4000 utterances
of the shuffled set go to the `valid` cutset and are not subject to speed
perturbation. The remaining utterances become the `train` cutset and are speed-
perturbed (0.9x, 1.0x, 1.1x).

"""

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RNG_SEED = 42


def make_cutset_blueprints(
    manifest_dir: Path,
    split: int,
) -> List[Tuple[str, CutSet]]:

    cut_sets = []
    # Create eval datasets
    logging.info("Creating eval cuts.")
    for i in range(1, 4):
        cut_set = CutSet.from_manifests(
            recordings=RecordingSet.from_file(
                manifest_dir / f"csj_recordings_eval{i}.jsonl.gz"
            ),
            supervisions=SupervisionSet.from_file(
                manifest_dir / f"csj_supervisions_eval{i}.jsonl.gz"
            ),
        )
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_sets.append((f"eval{i}", cut_set))

    # Create train and valid cuts
    logging.info("Loading, trimming, and shuffling the remaining core+noncore cuts.")
    recording_set = RecordingSet.from_file(
        manifest_dir / "csj_recordings_core.jsonl.gz"
    ) + RecordingSet.from_file(manifest_dir / "csj_recordings_noncore.jsonl.gz")
    supervision_set = SupervisionSet.from_file(
        manifest_dir / "csj_supervisions_core.jsonl.gz"
    ) + SupervisionSet.from_file(manifest_dir / "csj_supervisions_noncore.jsonl.gz")

    cut_set = CutSet.from_manifests(
        recordings=recording_set,
        supervisions=supervision_set,
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    cut_set = cut_set.shuffle(Random(RNG_SEED))

    logging.info(
        "Creating valid and train cuts from core and noncore, split at {split}."
    )
    valid_set = CutSet.from_cuts(islice(cut_set, 0, split))

    train_set = CutSet.from_cuts(islice(cut_set, split, None))
    train_set = train_set + train_set.perturb_speed(0.9) + train_set.perturb_speed(1.1)

    cut_sets.extend([("valid", valid_set), ("train", train_set)])

    return cut_sets


def get_args():
    parser = argparse.ArgumentParser(
        description=ARGPARSE_DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--manifest-dir", type=Path, help="Path to save manifests")
    parser.add_argument("--fbank-dir", type=Path, help="Path to save fbank features")
    parser.add_argument("--split", type=int, default=4000, help="Split at this index")

    return parser.parse_args()


def main():
    args = get_args()

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    num_jobs = min(16, os.cpu_count())

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    if (args.fbank_dir / ".done").exists():
        logging.info(
            "Previous fbank computed for CSJ found. "
            f"Delete {args.fbank_dir / '.done'} to allow recomputing fbank."
        )
        return
    else:
        cut_sets = make_cutset_blueprints(args.manifest_dir, args.split)
        for part, cut_set in cut_sets:
            logging.info(f"Processing {part}")
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                num_jobs=num_jobs,
                storage_path=(args.fbank_dir / f"feats_{part}").as_posix(),
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(args.manifest_dir / f"csj_cuts_{part}.jsonl.gz")

        logging.info("All fbank computed for CSJ.")
        (args.fbank_dir / ".done").touch()


if __name__ == "__main__":
    main()
