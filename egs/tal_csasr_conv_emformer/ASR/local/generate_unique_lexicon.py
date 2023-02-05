#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
This file takes as input a lexicon.txt and output a new lexicon,
in which each word has a unique pronunciation.

The way to do this is to keep only the first pronunciation of a word
in lexicon.txt.
"""


import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from icefall.lexicon import read_lexicon, write_lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        This file will generate a new file uniq_lexicon.txt
        in it.
        """,
    )

    return parser.parse_args()


def filter_multiple_pronunications(
    lexicon: List[Tuple[str, List[str]]]
) -> List[Tuple[str, List[str]]]:
    """Remove multiple pronunciations of words from a lexicon.

    If a word has more than one pronunciation in the lexicon, only
    the first one is kept, while other pronunciations are removed
    from the lexicon.

    Args:
      lexicon:
        The input lexicon, containing a list of (word, [p1, p2, ..., pn]),
        where "p1, p2, ..., pn" are the pronunciations of the "word".
    Returns:
      Return a new lexicon where each word has a unique pronunciation.
    """
    seen = set()
    ans = []

    for word, tokens in lexicon:
        if word in seen:
            continue
        seen.add(word)
        ans.append((word, tokens))
    return ans


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)

    lexicon_filename = lang_dir / "lexicon.txt"

    in_lexicon = read_lexicon(lexicon_filename)

    out_lexicon = filter_multiple_pronunications(in_lexicon)

    write_lexicon(lang_dir / "uniq_lexicon.txt", out_lexicon)

    logging.info(f"Number of entries in lexicon.txt: {len(in_lexicon)}")
    logging.info(f"Number of entries in uniq_lexicon.txt: {len(out_lexicon)}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
