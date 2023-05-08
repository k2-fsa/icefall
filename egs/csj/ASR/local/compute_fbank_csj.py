#!/usr/bin/env python3
# Copyright    2023  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
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
from pathlib import Path
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
from lhotse.recipes.csj import concat_csj_supervisions

# fmt: on

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RNG_SEED = 42
# concat_params_train = [
#     {"gap": 1.0, "maxlen": 10.0},
#     {"gap": 1.5, "maxlen": 8.0},
#     {"gap": 1.0, "maxlen": 18.0},
# ]

concat_params = {"gap": 1.0, "maxlen": 10.0}


def make_cutset_blueprints(
    manifest_dir: Path,
) -> List[Tuple[str, CutSet]]:

    cut_sets = []
    logging.info("Creating non-train cuts.")

    # Create eval datasets
    for i in range(1, 4):
        sps = sorted(
            SupervisionSet.from_file(
                manifest_dir / f"csj_supervisions_eval{i}.jsonl.gz"
            ),
            key=lambda x: x.id,
        )

        cut_set = CutSet.from_manifests(
            recordings=RecordingSet.from_file(
                manifest_dir / f"csj_recordings_eval{i}.jsonl.gz"
            ),
            supervisions=concat_csj_supervisions(sps, **concat_params),
        )
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_sets.append((f"eval{i}", cut_set))

    # Create excluded dataset
    sps = sorted(
        SupervisionSet.from_file(manifest_dir / "csj_supervisions_excluded.jsonl.gz"),
        key=lambda x: x.id,
    )
    cut_set = CutSet.from_manifests(
        recordings=RecordingSet.from_file(
            manifest_dir / "csj_recordings_excluded.jsonl.gz"
        ),
        supervisions=concat_csj_supervisions(sps, **concat_params),
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    cut_sets.append(("excluded", cut_set))

    # Create valid dataset
    sps = sorted(
        SupervisionSet.from_file(manifest_dir / "csj_supervisions_valid.jsonl.gz"),
        key=lambda x: x.id,
    )
    cut_set = CutSet.from_manifests(
        recordings=RecordingSet.from_file(
            manifest_dir / "csj_recordings_valid.jsonl.gz"
        ),
        supervisions=concat_csj_supervisions(sps, **concat_params),
    )
    cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
    cut_sets.append(("valid", cut_set))

    logging.info("Creating train cuts.")

    # Create train dataset
    sps = sorted(
        SupervisionSet.from_file(manifest_dir / "csj_supervisions_core.jsonl.gz")
        + SupervisionSet.from_file(manifest_dir / "csj_supervisions_noncore.jsonl.gz"),
        key=lambda x: x.id,
    )

    recording = RecordingSet.from_file(
        manifest_dir / "csj_recordings_core.jsonl.gz"
    ) + RecordingSet.from_file(manifest_dir / "csj_recordings_noncore.jsonl.gz")

    train_set = CutSet.from_manifests(
        recordings=recording, supervisions=concat_csj_supervisions(sps, **concat_params)
    ).trim_to_supervisions(keep_overlapping=False)
    train_set = train_set + train_set.perturb_speed(0.9) + train_set.perturb_speed(1.1)

    cut_sets.append(("train", train_set))

    return cut_sets


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--manifest-dir", type=Path, help="Path to save manifests"
    )
    parser.add_argument(
        "-f", "--fbank-dir", type=Path, help="Path to save fbank features"
    )

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
        cut_sets = make_cutset_blueprints(args.manifest_dir)
        for part, cut_set in cut_sets:
            logging.info(f"Processing {part}")
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                num_jobs=num_jobs,
                storage_path=(args.fbank_dir / f"feats_{part}").as_posix(),
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(args.fbank_dir / f"csj_cuts_{part}.jsonl.gz")

        logging.info("All fbank computed for CSJ.")
        (args.fbank_dir / ".done").touch()


if __name__ == "__main__":
    main()
