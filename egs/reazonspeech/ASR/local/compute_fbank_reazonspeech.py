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

# fmt: on

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RNG_SEED = 42
concat_params = {"gap": 1.0, "maxlen": 10.0}


def make_cutset_blueprints(
    manifest_dir: Path,
) -> List[Tuple[str, CutSet]]:
    cut_sets = []

    # Create test dataset
    logging.info("Creating test cuts.")
    cut_sets.append(
        (
            "test",
            CutSet.from_manifests(
                recordings=RecordingSet.from_file(
                    manifest_dir / "reazonspeech_recordings_test.jsonl.gz"
                ),
                supervisions=SupervisionSet.from_file(
                    manifest_dir / "reazonspeech_supervisions_test.jsonl.gz"
                ),
            ),
        )
    )

    # Create dev dataset
    logging.info("Creating dev cuts.")
    cut_sets.append(
        (
            "dev",
            CutSet.from_manifests(
                recordings=RecordingSet.from_file(
                    manifest_dir / "reazonspeech_recordings_dev.jsonl.gz"
                ),
                supervisions=SupervisionSet.from_file(
                    manifest_dir / "reazonspeech_supervisions_dev.jsonl.gz"
                ),
            ),
        )
    )

    # Create train dataset
    logging.info("Creating train cuts.")
    cut_sets.append(
        (
            "train",
            CutSet.from_manifests(
                recordings=RecordingSet.from_file(
                    manifest_dir / "reazonspeech_recordings_train.jsonl.gz"
                ),
                supervisions=SupervisionSet.from_file(
                    manifest_dir / "reazonspeech_supervisions_train.jsonl.gz"
                ),
            ),
        )
    )
    return cut_sets


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--manifest-dir", type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    num_jobs = min(16, os.cpu_count())

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    if (args.manifest_dir / ".reazonspeech-fbank.done").exists():
        logging.info(
            "Previous fbank computed for ReazonSpeech found. "
            f"Delete {args.manifest_dir / '.reazonspeech-fbank.done'} to allow recomputing fbank."
        )
        return
    else:
        cut_sets = make_cutset_blueprints(args.manifest_dir)
        for part, cut_set in cut_sets:
            logging.info(f"Processing {part}")
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                num_jobs=num_jobs,
                storage_path=(args.manifest_dir / f"feats_{part}").as_posix(),
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(args.manifest_dir / f"reazonspeech_cuts_{part}.jsonl.gz")

        logging.info("All fbank computed for ReazonSpeech.")
        (args.manifest_dir / ".reazonspeech-fbank.done").touch()


if __name__ == "__main__":
    main()
