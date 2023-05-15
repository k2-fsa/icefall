#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script takes the following two files as inputs:

- text
- recordings.jsonl.gz

and generates a file supervisions.jsonl.gz

## Motivation to have this file:

`lhotse kaldi import` requires that there has to be a file `segments`
in order to generate `supervisions.jsonl.gz`. However, dataset
like yesno from kaldi does not have such a file.

## Usage:

python3 ./text_to_supervision.py \
        --manifest-dir ./data/manifests \
        --text /path/to/text

"""

import argparse
import logging
from pathlib import Path

from lhotse import load_manifest
from lhotse.kaldi import load_kaldi_text_mapping
from lhotse.supervision import SupervisionSegment, SupervisionSet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="""We assume there is a file named recordings.jsonl.gz
        in this directory. We will save supervisions.jsonl.gz
        to this file
        """,
    )

    parser.add_argument(
        "--text",
        type=Path,
        required=True,
        help="Path to input text file",
    )

    return parser.parse_args()


def main():
    args = get_args()

    logging.info(f"{vars(args)}")

    if not (args.manifest_dir / "recordings.jsonl.gz").is_file():
        raise ValueError(f"{args.manifest_dir}/recordings.jsonl.gz does not exist")

    if not args.text.is_file():
        raise ValueError(f"{args.text} does not exist")

    recordings = load_manifest(args.manifest_dir / "recordings.jsonl.gz")

    text_dict = load_kaldi_text_mapping(args.text, must_exist=True)
    supervisions = []
    for utt_id, transcript in text_dict.items():
        if utt_id not in recordings:
            logging.info(f"{utt_id} does not exist in recordings. Skip it")
            continue
        supervisions.append(
            SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0,
                duration=recordings[utt_id].duration,
                channel=0,
                text=transcript,
            )
        )
    supervision_set = SupervisionSet.from_segments(supervisions)
    supervision_set.to_file(args.manifest_dir / "supervisions.jsonl.gz")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
