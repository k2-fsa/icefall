#!/usr/bin/env python3

"""
This script takes `bpe.model` as input and generates a file `tokens.txt`
from it.

Usage:
./bpe_model_to_tokens.py /path/to/input/bpe.model > tokens.txt
"""
import argparse

import sentencepiece as spm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bpe_model",
        type=str,
        help="Path to the input bpe.model",
    )

    return parser.parse_args()


def main():
    args = get_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    for i in range(sp.vocab_size()):
        print(sp.id_to_piece(i), i)


if __name__ == "__main__":
    main()
