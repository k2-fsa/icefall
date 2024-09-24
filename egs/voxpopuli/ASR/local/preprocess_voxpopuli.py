#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Yifan Yang)
#              2023  Brno University of Technology  (author: Karel Veselý)
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
Preprocess the database.
- Convert RecordingSet and SupervisionSet to CutSet.
- Apply text normalization to the transcripts.
   - We take renormalized `orig_text` as `text` transcripts.
   - The text normalization is separating punctuation from words.
   - Also we put capital letter to the beginning of a sentence.

The script is inspired in:
    `egs/commonvoice/ASR/local/preprocess_commonvoice.py`

Usage example:
    python3 ./local/preprocess_voxpopuli.py \
        --task asr --lang en

"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

# from local/
from separate_punctuation import separate_punctuation
from uppercase_begin_of_sentence import UpperCaseBeginOfSentence

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
        default=None,
    )

    parser.add_argument(
        "--task",
        type=str,
        help="""Task of VoxPopuli""",
        default="asr",
    )

    parser.add_argument(
        "--lang",
        type=str,
        help="""Language of VoxPopuli""",
        required=True,
    )

    parser.add_argument(
        "--use-original-text",
        type=str2bool,
        help="""Use 'original_text' from the annoattaion file,
                otherwise 'normed_text' will be used
                (see `data/manifests/${task}_${lang}.tsv.gz`).
             """,
        default=False,
    )

    return parser.parse_args()


def normalize_text(utt: str) -> str:
    utt = UpperCaseBeginOfSentence().process_line_text(separate_punctuation(utt))
    return utt


def preprocess_voxpopuli(
    task: str,
    language: str,
    dataset: Optional[str] = None,
    use_original_text: bool = False,
):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
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
    prefix = f"voxpopuli-{task}-{language}"
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

        if use_original_text:
            logging.info("Using 'original_text' from the annotation file.")
            logging.info(f"Normalizing text in {partition}")
            for sup in m["supervisions"]:
                # `orig_text` includes punctuation and true-case
                orig_text = str(sup.custom["orig_text"])
                # we replace `text` by normalized `orig_text`
                sup.text = normalize_text(orig_text)
        else:
            logging.info("Using 'normed_text' from the annotation file.")

        # remove supervisions with empty 'text'
        m["supervisions"] = m["supervisions"].filter(lambda sup: len(sup.text) > 0)

        # Create cut manifest with long-recordings.
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        ).resample(16000)

        # Store the cut set incl. the resampling.
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    preprocess_voxpopuli(
        task=args.task,
        language=args.lang,
        dataset=args.dataset,
        use_original_text=args.use_original_text,
    )
    logging.info("Done")


if __name__ == "__main__":
    main()
