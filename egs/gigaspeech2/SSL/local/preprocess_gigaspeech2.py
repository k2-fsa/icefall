#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.             (Yifan Yang)
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

import logging
import re
from pathlib import Path
import unicodedata

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        type=str,
    )

    return parser.parse_args()


def normalize_text(
    text: str,
    lang: str,
) -> str:
    text = unicodedata.normalize("NFKC", text)

    # Convert to upper case
    text = text.upper()

    # Remove brackets with content
    text = re.sub(r"\([^\)]*\)", " ", text)

    # Language-related normalization
    if lang == "Thai":
        # Digit mapping
        text = re.sub(r"\u0030", r"\u0E50", text)
        text = re.sub(r"\u0031", r"\u0E51", text)
        text = re.sub(r"\u0032", r"\u0E52", text)
        text = re.sub(r"\u0033", r"\u0E53", text)
        text = re.sub(r"\u0034", r"\u0E54", text)
        text = re.sub(r"\u0035", r"\u0E55", text)
        text = re.sub(r"\u0036", r"\u0E56", text)
        text = re.sub(r"\u0037", r"\u0E57", text)
        text = re.sub(r"\u0038", r"\u0E58", text)
        text = re.sub(r"\u0039", r"\u0E59", text)

        # Currency symbols mapping
        text = re.sub(r"\u0024", "ดอลลาร์", text)  # $
        text = re.sub(r"\u00A3", "ปอนด์", text)  # £
        text = re.sub(r"\u00A5", "หยวน", text)  # ¥
        text = re.sub(r"\u20AC", "ยูโร", text)  # €
        text = re.sub(r"\u0E3F", "บาท", text)  # ฿

        # Temperature/Angle symbols mapping
        text = re.sub(r"\u00B0\u0043", "องศาเซลเซียส", text)  # °C
        text = re.sub(r"\u00B0\u0046", "องศาฟาเรนไฮต์", text)  # °F
        text = re.sub(r"\u00B0", "องศา", text)  # °
    
        # Remove blank symbols
        text = re.sub(r"\s", "", utt)

    else:
        text = re.sub(r"\s+", " ", text).strip()

    return utt


def preprocess_gigaspeech2(args):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = ("test",)

    logging.info("Loading manifest (may take 4 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix="gigaspeech2",
        suffix="jsonl.gz",
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
        raw_cuts_path = output_dir / f"gigaspeech2_cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        for sup in m["supervisions"]:
            sup.text = normalize_text(sup.text)

        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    preprocess_gigaspeech2(args)


if __name__ == "__main__":
    main()
