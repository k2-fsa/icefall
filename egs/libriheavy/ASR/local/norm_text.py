#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
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
import codecs
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        help="""Path to the input text.
        """,
    )
    return parser.parse_args()


def remove_punc_to_upper(text: str) -> str:
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    tokens = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'")
    s_list = [x.upper() if x in tokens else " " for x in text]
    s = " ".join("".join(s_list).split()).strip()
    return s


def main():
    args = get_args()
    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    line = f.readline()
    while line:
        print(remove_punc_to_upper(line))
        line = f.readline()


if __name__ == "__main__":
    main()
