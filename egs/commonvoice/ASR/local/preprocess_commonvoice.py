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

    parser.add_argument(
        "--language",
        type=str,
        help="""Language of Common Voice""",
    )

    return parser.parse_args()


def normalize_text(utt: str, language: str) -> str:
    utt = re.sub(r"[{0}]+".format("-"), " ", utt)
    utt = re.sub("’", "'", utt)
    if language == "en":
        return re.sub(r"[^a-zA-Z\s]", "", utt).upper()
    if language == "fr":
        return re.sub(r"[^A-ZÀÂÆÇÉÈÊËÎÏÔŒÙÛÜ' ]", "", utt).upper()


def preprocess_commonvoice(
    language: str,
    dataset: Optional[str] = None,
):
    src_dir = Path(f"data/{language}/manifests")
    output_dir = Path(f"data/{language}/fbank")
    output_dir.mkdir(exist_ok=True)

    if dataset is None:
        dataset_parts = (
            "dev",
            "test",
            "train",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    logging.info("Loading manifest")
    prefix = f"cv-{language}"
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
        for sup in m["supervisions"]:
            text = str(sup.text)
            orig_text = text
            sup.text = normalize_text(sup.text, language)
            text = str(sup.text)
            if len(orig_text) != len(text):
                logging.info(
                    f"\nOriginal text vs normalized text:\n{orig_text}\n{text}"
                )

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
    preprocess_commonvoice(
        language=args.language,
        dataset=args.dataset,
    )
    logging.info("Done")


if __name__ == "__main__":
    main()
