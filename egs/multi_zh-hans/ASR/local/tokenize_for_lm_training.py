#!/usr/bin/env python3
# Copyright    2017  Johns Hopkins University   (authors: Shinji Watanabe)
#              2022  Xiaomi Corp.               (authors: Mingshuang Luo)
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
import re
import sys
from typing import List

from pypinyin import lazy_pinyin, pinyin
from icefall.utils import str2bool, tokenize_by_CJK_char

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert raw text to tokenized text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-ncols", "-s", default=0, type=int, help="skip first n columns"
    )
    parser.add_argument(
        "--non-lang-syms",
        "-l",
        default=None,
        type=str,
        help="list of non-linguistic symobles, e.g., <NOISE> etc.",
    )
    parser.add_argument("text", type=str, default=False, nargs="?", help="input text")
    parser.add_argument(
        "--trans_type",
        "-t",
        type=str,
        default="char",
        choices=["char", "pinyin", "lazy_pinyin"],
        help="""Transcript type. char/pinyin/lazy_pinyin""",
    )
    return parser


def token2id(
    texts, token_table, token_type: str = "lazy_pinyin", oov: str = "<unk>"
) -> List[List[int]]:
    """Convert token to id.
    Args:
      texts:
        The input texts, it refers to the chinese text here.
      token_table:
        The token table is built based on "data/lang_xxx/token.txt"
      token_type:
        The type of token, such as "pinyin" and "lazy_pinyin".
      oov:
        Out of vocabulary token. When a word(token) in the transcript
        does not exist in the token list, it is replaced with `oov`.

    Returns:
      The list of ids for the input texts.
    """
    if texts is None:
        raise ValueError("texts can't be None!")
    else:
        oov_id = token_table[oov]
        ids: List[List[int]] = []
        for text in texts:
            chars_list = list(str(text))
            if token_type == "lazy_pinyin":
                text = lazy_pinyin(chars_list)
                sub_ids = [
                    token_table[txt] if txt in token_table else oov_id for txt in text
                ]
                ids.append(sub_ids)
            else:  # token_type = "pinyin"
                text = pinyin(chars_list)
                sub_ids = [
                    token_table[txt[0]] if txt[0] in token_table else oov_id
                    for txt in text
                ]
                ids.append(sub_ids)
        return ids


def main():
    parser = get_parser()
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with codecs.open(args.non_lang_syms, "r", encoding="utf-8") as f:
            nls = [x.rstrip() for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(
        sys.stdout if is_python2 else sys.stdout.buffer
    )
    line = f.readline()
    while line:
        x = line.split()
        print(" ".join(x[: args.skip_ncols]), end=" ")
        a = " ".join(x[args.skip_ncols :])  # noqa E203

        a_flat = tokenize_by_CJK_char(a)

        # print("".join(a_chars))
        print(a_flat)
        line = f.readline()


if __name__ == "__main__":
    main()
