#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Yifan Yang)
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
import re
from pathlib import Path
from typing import Optional

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    return parser.parse_args()


def normalize_text(utt: str) -> str:
    utt = re.sub(r"[{0}]+".format("-"), " ", utt)
    return re.sub(r"[^a-zA-Z\s]", "", utt).upper()


def preprocess_peoples_speech(dataset: Optional[str] = None):
    src_dir = Path(f"data/manifests")
    output_dir = Path(f"data/fbank")
    output_dir.mkdir(exist_ok=True)

    if dataset is None:
        dataset_parts = (
            "validation",
            "test",
            "dirty",
            "dirty_sa",
            "clean",
            "clean_sa",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    logging.info("Loading manifest, it may takes 8 minutes")
    prefix = f"peoples_speech"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix=suffix,
        prefix=prefix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"{prefix}_cuts_{partition}_raw.{suffix}"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        logging.info(f"Normalizing text in {partition}")
        i = 0
        for sup in m["supervisions"]:
            text = str(sup.text)
            orig_text = text
            sup.text = normalize_text(sup.text)
            text = str(sup.text)
            if i < 10 and len(orig_text) != len(text):
                logging.info(
                    f"\nOriginal text vs normalized text:\n{orig_text}\n{text}"
                )
                i += 1

        # Create long-recording cut manifests.
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        ).resample(16000)

        # Run data augmentation that needs to be done in the
        # time domain.
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    preprocess_peoples_speech(dataset=args.dataset)
    logging.info("Done")


if __name__ == "__main__":
    main()
