#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Zengwei Yao,
#                                                       Zengrui Jin,
#                                                       Wei Kang)
#                   2024  Tsinghua University (authors: Zengrui Jin,)
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
This file reads the texts in given manifest and save the cleaned new cuts.
"""

import argparse
import logging
import glob
import os
from pathlib import Path
from typing import List

from lhotse import CutSet, load_manifest_lazy
from concurrent.futures import ProcessPoolExecutor as Pool

from tokenizer import (
    is_alphabet,
    is_chinese,
    is_hangul,
    is_japanese,
    tokenize_by_CJK_char,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subset",
        type=str,
        help="Subset of emilia, (ZH, EN, etc.)",
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=20,
        help="Number of jobs to processing.",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/manifests/splits_raw",
        help="The source directory of manifest files.",
    )

    parser.add_argument(
        "--dest-dir",
        type=str,
        default="data/manifests/splits",
        help="The destination directory of manifest files.",
    )

    return parser.parse_args()


def preprocess_emilia(file_name: str, input_dir: Path, output_dir: Path):
    logging.info(f"Processing {file_name}")
    if (output_dir / file_name).is_file():
        logging.info(f"{file_name} exists, skipping.")
        return

    def _filter_cut(cut):
        text = cut.supervisions[0].text
        duration = cut.supervisions[0].duration
        chinese = []
        english = []

        # only contains chinese and space and alphabets
        clean_chars = []
        for x in text:
            if is_hangul(x):
                logging.warning(f"Delete cut with text containing Korean : {text}")
                return False
            if is_japanese(x):
                logging.warning(f"Delete cut with text containing Japanese : {text}")
                return False
            if is_chinese(x):
                chinese.append(x)
                clean_chars.append(x)
            if is_alphabet(x):
                english.append(x)
                clean_chars.append(x)
            if x == " ":
                clean_chars.append(x)
        if len(english) + len(chinese) == 0:
            logging.warning(f"Delete cut with text has no valid chars : {text}")
            return False

        words = tokenize_by_CJK_char("".join(clean_chars))
        for i in range(len(words) - 10):
            if words[i : i + 10].count(words[i]) == 10:
                logging.warning(f"Delete cut with text with too much repeats : {text}")
                return False
        # word speed, 20 - 600 / minute
        if duration < len(words) / 600 * 60 or duration > len(words) / 20 * 60:
            logging.warning(
                f"Delete cut with audio text mismatch, duration : {duration}s, "
                f"words : {len(words)}, text : {text}"
            )
            return False
        return True

    try:
        cut_set = load_manifest_lazy(input_dir / file_name)
        cut_set = cut_set.filter(_filter_cut)
        cut_set.to_file(output_dir / file_name)
    except Exception as e:
        logging.error(f"Manifest {file_name} failed with error: {e}")
        os.remove(str(output_dir / file_name))


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()

    input_dir = Path(args.source_dir)
    output_dir = Path(args.dest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cut_files = glob.glob(f"{args.source_dir}/emilia_cuts_{args.subset}.*.jsonl.gz")

    with Pool(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                preprocess_emilia, filename.split("/")[-1], input_dir, output_dir
            )
            for filename in cut_files
        ]
        for f in futures:
            f.result()
            f.done()
    logging.info("Processing done.")
