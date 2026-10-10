#!/usr/bin/env python3

"""
This file generates the file lexicon.txt that contains pronunciations of all
words and phrases
"""

from pypinyin import phrases_dict, pinyin_dict
from tokenizer import Tokenizer

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/lexicon.txt",
        help="""Path to save the generated lexicon.""",
    )
    return parser


def main():
    args = get_parser().parse_args()
    filename = args.lexicon
    tokens = args.tokens
    tokenizer = Tokenizer(tokens)

    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    i = 0
    with open(filename, "w", encoding="utf-8") as f:
        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FFF):
                continue

            w = chr(key)

            # 1 to remove the initial sil
            # :-1 to remove the final eos
            tokens = tokenizer.text_to_tokens(w)[1:-1]

            tokens = " ".join(tokens)
            f.write(f"{w} {tokens}\n")

        # TODO(fangjun): Add phrases
        #  for key in phrases:
        #      # 1 to remove the initial sil
        #      # :-1 to remove the final eos
        #      tokens = tokenizer.text_to_tokens(key)[1:-1]
        #      tokens = " ".join(tokens)
        #      f.write(f"{key} {tokens}\n")


if __name__ == "__main__":
    main()
