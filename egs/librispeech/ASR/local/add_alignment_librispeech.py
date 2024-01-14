#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Zengwei Yao)
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
This file adds alignments from https://github.com/CorentinJ/librispeech-alignments  # noqa
to the existing fbank features dir (e.g., data/fbank)
and save cuts to a new dir (e.g., data/fbank_ali).
"""

import argparse
import logging
import zipfile
from pathlib import Path
from typing import List

from lhotse import CutSet, load_manifest_lazy
from lhotse.recipes.librispeech import parse_alignments
from lhotse.utils import is_module_available

LIBRISPEECH_ALIGNMENTS_URL = (
    "https://drive.google.com/uc?id=1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE"
)

DATASET_PARTS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--alignments-dir",
        type=str,
        default="data/alignment",
        help="The dir to save alignments.",
    )

    parser.add_argument(
        "--cuts-in-dir",
        type=str,
        default="data/fbank",
        help="The dir of the existing cuts without alignments.",
    )

    parser.add_argument(
        "--cuts-out-dir",
        type=str,
        default="data/fbank_ali",
        help="The dir to save the new cuts with alignments",
    )

    return parser


def download_alignments(
    target_dir: str, alignments_url: str = LIBRISPEECH_ALIGNMENTS_URL
):
    """
    Download and extract the alignments.

    Note: If you can not access drive.google.com, you could download the file
    `LibriSpeech-Alignments.zip` from huggingface:
    https://huggingface.co/Zengwei/librispeech-alignments
    and extract the zip file manually.

    Args:
      target_dir:
        The dir to save alignments.
      alignments_url:
        The URL of alignments.
    """
    """Modified from https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/librispeech.py"""  # noqa
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    completed_detector = target_dir / ".ali_completed"
    if completed_detector.is_file():
        logging.info("The alignment files already exist.")
        return

    ali_zip_path = target_dir / "LibriSpeech-Alignments.zip"
    if not ali_zip_path.is_file():
        assert is_module_available(
            "gdown"
        ), 'To download LibriSpeech alignments, please install "pip install gdown"'  # noqa
        import gdown

        gdown.download(alignments_url, output=str(ali_zip_path))

    with zipfile.ZipFile(str(ali_zip_path)) as f:
        f.extractall(path=target_dir)
        completed_detector.touch()


def add_alignment(
    alignments_dir: str,
    cuts_in_dir: str = "data/fbank",
    cuts_out_dir: str = "data/fbank_ali",
    dataset_parts: List[str] = DATASET_PARTS,
):
    """
    Add alignment info to existing cuts.

    Args:
      alignments_dir:
        The dir of the alignments.
      cuts_in_dir:
        The dir of the existing cuts.
      cuts_out_dir:
        The dir to save the new cuts with alignments.
      dataset_parts:
        Librispeech parts to add alignments.
    """
    alignments_dir = Path(alignments_dir)
    cuts_in_dir = Path(cuts_in_dir)
    cuts_out_dir = Path(cuts_out_dir)
    cuts_out_dir.mkdir(parents=True, exist_ok=True)

    for part in dataset_parts:
        logging.info(f"Processing {part}")

        cuts_in_path = cuts_in_dir / f"librispeech_cuts_{part}.jsonl.gz"
        if not cuts_in_path.is_file():
            logging.info(f"{cuts_in_path} does not exist - skipping.")
            continue
        cuts_out_path = cuts_out_dir / f"librispeech_cuts_{part}.jsonl.gz"
        if cuts_out_path.is_file():
            logging.info(f"{part} already exists - skipping.")
            continue

        # parse alignments
        alignments = {}
        part_ali_dir = alignments_dir / "LibriSpeech" / part
        for ali_path in part_ali_dir.rglob("*.alignment.txt"):
            ali = parse_alignments(ali_path)
            alignments.update(ali)
        logging.info(f"{part} has {len(alignments.keys())} cuts with alignments.")

        # add alignment attribute and write out
        cuts_in = load_manifest_lazy(cuts_in_path)
        with CutSet.open_writer(cuts_out_path) as writer:
            for cut in cuts_in:
                for idx, subcut in enumerate(cut.supervisions):
                    origin_id = subcut.id.split("_")[0]
                    if origin_id in alignments:
                        ali = alignments[origin_id]
                    else:
                        logging.info(f"Warning: {origin_id} does not have alignment.")
                        ali = []
                    subcut.alignment = {"word": ali}
                writer.write(cut, flush=True)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    download_alignments(args.alignments_dir)
    add_alignment(args.alignments_dir, args.cuts_in_dir, args.cuts_out_dir)


if __name__ == "__main__":
    main()
