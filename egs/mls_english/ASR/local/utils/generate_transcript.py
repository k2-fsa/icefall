#!/usr/bin/env python3
# Copyright    2022  The University of Electro-Communications  (Author: Teo Wen Shen)  # noqa
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
from pathlib import Path
from typing import Optional

from lhotse import CutSet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate transcripts for BPE training from MLS English dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="parler-tts/mls_eng",
        help="Path to HuggingFace MLS English dataset (name or local path)",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang"),
        help="Directory to store output transcripts",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for generating transcripts (train/dev/test)",
    )

    return parser.parse_args()


def generate_transcript_from_cuts(cuts: CutSet, output_file: Path) -> None:
    """Generate transcript text file from Lhotse CutSet."""
    with open(output_file, "w") as f:
        for cut in tqdm(cuts, desc="Processing cuts"):
            for sup in cut.supervisions:
                f.write(f"{sup.text}\n")


def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args.lang_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.lang_dir / "transcript.txt"

    logging.info(f"Loading {args.split} split from dataset: {args.dataset_path}")
    try:
        cuts = CutSet.from_huggingface_dataset(
            args.dataset_path, split=args.split, text_key="transcript"
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

    logging.info(f"Generating transcript to {output_file}")
    generate_transcript_from_cuts(cuts, output_file)
    logging.info("Transcript generation completed")


if __name__ == "__main__":
    main()
