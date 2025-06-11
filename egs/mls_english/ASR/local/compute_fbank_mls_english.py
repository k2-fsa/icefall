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
from lhotse.utils import is_module_available

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
    mls_eng_hf_dataset_path: str = "parler-tts/mls_eng",
) -> List[Tuple[str, CutSet]]:
    cut_sets = []

    if not is_module_available("datasets"):
        raise ImportError(
            "To process the MLS English HF corpus, please install optional dependency: pip install datasets"
        )

    from datasets import load_dataset

    print(f"{mls_eng_hf_dataset_path=}")
    dataset = load_dataset(str(mls_eng_hf_dataset_path))

    # Create test dataset
    logging.info("Creating test cuts.")
    cut_sets.append(
        (
            "test",
            CutSet.from_huggingface_dataset(dataset["test"], text_key="transcript"),
        )
    )

    # Create dev dataset
    logging.info("Creating dev cuts.")
    try:
        cut_sets.append(
            ("dev", CutSet.from_huggingface_dataset(dataset["dev"], text_key="transcript"))
        )
    except KeyError:
        cut_sets.append(
            ("dev", CutSet.from_huggingface_dataset(dataset["validation"], text_key="transcript"))
        )

    # Create train dataset
    logging.info("Creating train cuts.")
    cut_sets.append(
        (
            "train",
            CutSet.from_huggingface_dataset(dataset["train"], text_key="transcript"),
        )
    )
    return cut_sets


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--manifest-dir", type=Path)
    parser.add_argument("-a", "--audio-dir", type=Path)
    parser.add_argument("-d", "--dl-dir", type=Path)
    return parser.parse_args()


def main():
    args = get_args()

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    num_jobs = min(16, os.cpu_count())

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    if (args.manifest_dir / ".mls-eng-fbank.done").exists():
        logging.info(
            "Previous fbank computed for MLS English found. "
            f"Delete {args.manifest_dir / '.mls-eng-fbank.done'} to allow recomputing fbank."
        )
        return
    else:
        mls_eng_hf_dataset_path = args.dl_dir # "/root/datasets/parler-tts--mls_eng"
        cut_sets = make_cutset_blueprints(mls_eng_hf_dataset_path)
        for part, cut_set in cut_sets:
            logging.info(f"Processing {part}")
            cut_set = cut_set.save_audios(
                num_jobs=num_jobs,
                storage_path=(args.audio_dir / part).as_posix(),
            ) # makes new cutset that loads audio from paths to actual audio files
            
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                num_jobs=num_jobs,
                storage_path=(args.manifest_dir / f"feats_{part}").as_posix(),
                storage_type=LilcomChunkyWriter,
            )

            cut_set.to_file(args.manifest_dir / f"mls_eng_cuts_{part}.jsonl.gz")

        logging.info("All fbank computed for MLS English.")
        (args.manifest_dir / ".mls-eng-fbank.done").touch()


if __name__ == "__main__":
    main()
