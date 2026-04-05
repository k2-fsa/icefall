#!/usr/bin/env python3

"""
This file generates the file tokens.txt.

Usage:

python3 ./local/generate_tokens.py > data/tokens.txt
"""


import argparse
from typing import List

import jieba
from pypinyin import Style, lazy_pinyin, pinyin_dict


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
        help="Path to to save tokens.txt.",
    )

    return parser


def generate_token_list() -> List[str]:
    token_set = set()

    word_dict = pinyin_dict.pinyin_dict
    i = 0
    for key in word_dict:
        if not (0x4E00 <= key <= 0x9FFF):
            continue

        w = chr(key)
        t = lazy_pinyin(w, style=Style.TONE3, tone_sandhi=True)[0]
        token_set.add(t)

    no_digit = set()
    for t in token_set:
        if t[-1] not in "1234":
            no_digit.add(t)
        else:
            no_digit.add(t[:-1])

    no_digit.add("dei")
    no_digit.add("tou")
    no_digit.add("dia")

    for t in no_digit:
        token_set.add(t)
        for i in range(1, 5):
            token_set.add(f"{t}{i}")

    ans = list(token_set)
    ans.sort()

    punctuations = list(",.!?:\"'")
    ans = punctuations + ans

    # use ID 0 for blank
    # Use ID 1 of _ for padding
    ans.insert(0, " ")
    ans.insert(1, "_")  #

    return ans


def main():
    args = get_parser().parse_args()
    token_list = generate_token_list()
    with open(args.tokens, "w", encoding="utf-8") as f:
        for indx, token in enumerate(token_list):
            f.write(f"{token} {indx}\n")


if __name__ == "__main__":
    main()
