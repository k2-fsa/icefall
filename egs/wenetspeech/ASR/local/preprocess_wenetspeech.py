#!/usr/bin/env python3
# Copyright    2021  Johns Hopkins University (Piotr Żelasko)
# Copyright    2021  Xiaomi Corp.             (Fangjun Kuang)
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

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall import setup_logger

# Similar text filtering and normalization procedure as in:
# https://github.com/SpeechColab/WenetSpeech/blob/main/toolkits/kaldi/wenetspeech_data_prep.sh


def normalize_text(
    utt: str,
    # punct_pattern=re.compile(r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    punct_pattern=re.compile(r"<(PERIOD|QUESTIONMARK|EXCLAMATIONPOINT)>"),
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


def has_no_oov(
    sup: SupervisionSegment,
    oov_pattern=re.compile(r"<(SIL|MUSIC|NOISE|OTHER)>"),
) -> bool:
    return oov_pattern.search(sup.text) is None


def preprocess_wenet_speech():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    # Note: By default, we preprocess all sub-parts.
    # You can delete those that you don't need.
    # For instance, if you don't want to use the L subpart, just remove
    # the line below containing "L"
    dataset_parts = (
        "DEV",
        "TEST_NET",
        "TEST_MEETING",
        "S",
        "M",
        "L",
    )

    logging.info("Loading manifest (may take 10 minutes)")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix="jsonl.gz",
        prefix="wenetspeech",
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
        raw_cuts_path = output_dir / f"cuts_{partition}_raw.jsonl.gz"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        # Note this step makes the recipe different than LibriSpeech:
        # We must filter out some utterances and remove punctuation
        # to be consistent with Kaldi.
        logging.info("Filtering OOV utterances from supervisions")
        m["supervisions"] = m["supervisions"].filter(has_no_oov)
        logging.info(f"Normalizing text in {partition}")
        for sup in m["supervisions"]:
            text = str(sup.text)
            orig_text = text
            sup.text = normalize_text(sup.text)
            text = str(sup.text)
            if len(orig_text) != len(text):
                logging.info(
                    f"\nOriginal text vs normalized text:\n{orig_text}\n{text}"
                )

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        if partition not in ["DEV", "TEST_NET", "TEST_MEETING"]:
            logging.info(
                f"Speed perturb for {partition} with factors 0.9 and 1.1 "
                "(Perturbing may take 8 minutes and saving may take 20 minutes)"
            )
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    setup_logger(log_filename="./log-preprocess-wenetspeech")

    preprocess_wenet_speech()
    logging.info("Done")


if __name__ == "__main__":
    main()
