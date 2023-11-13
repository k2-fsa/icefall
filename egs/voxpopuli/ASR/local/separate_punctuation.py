#!/usr/bin/env python3
# Copyright    2023  Brno University of Technology  (authors: Karel Veselý)
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
This script chops the punctuation as standalone tokens.
Example:
    input: "This is fine. Yes, you are right."
    output: "This is fine . Yes , you are right ."

The script also handles exceptions in a hard-coded fashion.

(same functionality could be done with `nltk.tokenize.word_tokenize()`,
 but that would be an extra dependency)

It can be used as a module, or as an executable script.

Usage example #1:
  `from separate_punctuation import separate_punctuation`

Usage example #2:
```
  python3 ./local/separate_punctuation.py \
    --ignore-columns 1 \
    < ${kaldi_data}/text
```
"""

import re
import sys
from argparse import ArgumentParser


def separate_punctuation(text: str) -> str:
    """
    Text filtering function for separating punctuation.

    Example:
        input: "This is fine. Yes, you are right."
        output: "This is fine . Yes , you are right ."

    The exceptions for which the punctuation is
    not splitted are hard-coded.
    """

    # remove non-desired punctuation symbols
    text = re.sub('["„“«»]', "", text)

    # separate [,.!?;] punctuation from words by space
    text = re.sub(r"(\w)([,.!?;])", r"\1 \2", text)
    text = re.sub(r"([,.!?;])(\w)", r"\1 \2", text)

    # split to tokens
    tokens = text.split()
    tokens_out = []

    # re-join the special cases of punctuation
    for ii, tok in enumerate(tokens):
        # no rewriting for 1st and last token
        if ii > 0 and ii < len(tokens) - 1:
            # **RULES ADDED FOR CZECH COMMON VOICE**

            # fix "27 . dubna" -> "27. dubna", but keep punctuation separate,
            if tok == "." and tokens[ii - 1].isdigit() and tokens[ii + 1].islower():
                tokens_out[-1] = tokens_out[-1] + "."
                continue

            # fix "resp . pak" -> "resp. pak"
            if tok == "." and tokens[ii - 1].isalpha() and tokens[ii + 1].islower():
                tokens_out[-1] = tokens_out[-1] + "."
                continue

            # **RULES ADDED FOR ENGLISH COMMON VOICE**

            # fix "A ." -> "A."
            if tok == "." and re.match(r"^[A-Z]S", tokens[ii - 1]):
                tokens_out[-1] = tokens_out[-1] + "."
                continue

            # fix "Mr ." -> "Mr."
            exceptions = set(["Mr", "Mrs", "Ms"])
            if tok == "." and tokens[ii - 1] in exceptions:
                tokens_out[-1] = tokens_out[-1] + "."
                continue

        tokens_out.append(tok)

    return " ".join(tokens_out)


def get_args():
    parser = ArgumentParser(
        description="Separate punctuation from words: 'hello.' -> 'hello .'"
    )
    parser.add_argument(
        "--ignore-columns", type=int, default=1, help="skip number of initial columns"
    )
    return parser.parse_args()


def main():
    args = get_args()

    max_split = args.ignore_columns

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        *key, text = line.strip().split(maxsplit=max_split)
        text_norm = separate_punctuation(text)

        print(" ".join(key), text_norm)


if __name__ == "__main__":
    main()
