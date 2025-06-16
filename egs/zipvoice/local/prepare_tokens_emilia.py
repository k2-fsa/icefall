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
This file reads the texts in given manifest and save the new cuts with phoneme tokens.
"""

import argparse
import glob
import logging
import re
from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path
from typing import List

import jieba
from lhotse import load_manifest_lazy
from tokenizer import Tokenizer, is_alphabet, is_chinese, is_hangul, is_japanese


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
        default=50,
        help="Number of jobs to processing.",
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/manifests_emilia/splits",
        help="The source directory of manifest files.",
    )

    parser.add_argument(
        "--dest-dir",
        type=str,
        help="The destination directory of manifest files.",
    )

    return parser.parse_args()


def tokenize_by_CJK_char(line: str) -> List[str]:
    """
    Tokenize a line of text with CJK char.

    Note: All return characters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = [你, 好, 世, 界, 是, HELLO, WORLD, 的, 中, 文]

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip().upper())
    char_list = []
    for w in chars:
        if w.strip():
            char_list += w.strip().split()
    return char_list


def prepare_tokens_emilia(file_name: str, input_dir: Path, output_dir: Path):
    logging.info(f"Processing {file_name}")
    if (output_dir / file_name).is_file():
        logging.info(f"{file_name} exists, skipping.")
        return
    jieba.setLogLevel(logging.INFO)
    tokenizer = Tokenizer()

    def _prepare_cut(cut):
        # Each cut only contains one supervision
        assert len(cut.supervisions) == 1, (len(cut.supervisions), cut)
        text = cut.supervisions[0].text
        cut.supervisions[0].normalized_text = text
        tokens = tokenizer.texts_to_tokens([text])[0]
        cut.tokens = tokens
        return cut

    def _filter_cut(cut):
        text = cut.supervisions[0].text
        duration = cut.supervisions[0].duration
        chinese = []
        english = []

        # only contains chinese and space and alphabets
        clean_chars = []
        for x in text:
            if is_hangul(x):
                logging.info(f"Delete cut with text containing Korean : {text}")
                return False
            if is_japanese(x):
                logging.info(f"Delete cut with text containing Japanese : {text}")
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
            logging.info(f"Delete cut with text has no valid chars : {text}")
            return False

        words = tokenize_by_CJK_char("".join(clean_chars))
        for i in range(len(words) - 10):
            if words[i : i + 10].count(words[i]) == 10:
                logging.info(f"Delete cut with text with too much repeats : {text}")
                return False
        # word speed, 20 - 600 / minute
        if duration < len(words) / 600 * 60 or duration > len(words) / 20 * 60:
            logging.info(
                f"Delete cut with audio text mismatch, duration : {duration}s, words : {len(words)}, text : {text}"
            )
            return False
        return True

    try:
        cut_set = load_manifest_lazy(input_dir / file_name)
        cut_set = cut_set.filter(_filter_cut)
        cut_set = cut_set.map(_prepare_cut)
        cut_set.to_file(output_dir / file_name)
    except Exception as e:
        logging.error(f"Manifest {file_name} failed with error: {e}")
        raise


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
                prepare_tokens_emilia, filename.split("/")[-1], input_dir, output_dir
            )
            for filename in cut_files
        ]
        for f in futures:
            try:
                f.result()
                f.done()
            except Exception as e:
                logging.error(f"Future failed with error: {e}")
    logging.info("Processing done.")
