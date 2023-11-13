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
This script introduces initial capital letter at the beginning of a sentence.
It can be used as a module, or as an executable script.

Usage example #1:
  `from uppercase_begin_of_sentence import UpperCaseBeginOfSentence`

Usage example #2:
```
  python3 ./local/uppercase_begin_of_sentence.py \
    --ignore-columns 1 \
    < ${kaldi_data}/text
```
"""

import re
import sys
from argparse import ArgumentParser


class UpperCaseBeginOfSentence:
    """
    This class introduces initial capital letter at the beginning of a sentence.
    Capital letter is used, if previous symbol was punctuation token from
    `set([".", "!", "?"])`.

    The punctuation as previous token is memorized also across
    `process_line_text()` calls.
    """

    def __init__(self):
        # The 1st word will have Title-case
        # This variable transfers context from previous line
        self.prev_token_is_punct = True

    def process_line_text(self, line_text: str) -> str:
        """
        It is assumed that punctuation in `line_text` was already separated,
        example: "This is fine . Yes , you are right ."
        """

        words = line_text.split()
        punct_set = set([".", "!", "?"])

        for ii, w in enumerate(words):
            # punctuation ?
            if w in punct_set:
                self.prev_token_is_punct = True
                continue

            # change case of word...
            if self.prev_token_is_punct:
                if re.match("<", w):
                    continue  # skip <symbols>
                # apply Title-case only on lowercase words.
                if w.islower():
                    words[ii] = w.title()
                # change state
                self.prev_token_is_punct = False

        line_text_uc = " ".join(words)

        return line_text_uc


def get_args():
    parser = ArgumentParser(
        description="Put upper-case at the beginning of a sentence."
    )
    parser.add_argument(
        "--ignore-columns", type=int, default=4, help="skip number of initial columns"
    )
    return parser.parse_args()


def main():
    args = get_args()

    uc_bos = UpperCaseBeginOfSentence()
    max_split = args.ignore_columns

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()

        if len(line.split()) > 1:
            *key, text = line.strip().split(maxsplit=max_split)  # parse,
            text_uc = uc_bos.process_line_text(text)  # process,
            print(" ".join(key), text_uc)  # print,
        else:
            print(line)


if __name__ == "__main__":
    main()
