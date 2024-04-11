#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
This file generates words.txt from the given transcript file.
"""

import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "transcript",
        type=Path,
        help="Input transcript file",
    )
    return parser.parse_args()


def main():
    args = get_args()
    assert args.transcript.is_file(), args.transcript

    word_set = set()
    with open(args.transcript) as f:
        for line in f:
            words = line.strip().split()
            for w in words:
                word_set.add(w)

    # Note: reserved* should be kept in sync with ./local/prepare_lang_bpe.py
    reserved1 = ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>"]
    reserved2 = ["#0", "<s>", "</s>"]

    for w in reserved1 + reserved2:
        assert w not in word_set, w

    words = sorted(list(word_set))
    words = reserved1 + words + reserved2

    for i, w in enumerate(words):
        print(w, i)


if __name__ == "__main__":
    main()
