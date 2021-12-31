#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Pingfeng Luo)
import argparse
import re
from pathlib import Path
from typing import Dict, List
from pypinyin import pinyin, lazy_pinyin, Style

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", type=str, help="The input lexicon file.")
    return parser.parse_args()


def process_line(
    line: str
) -> None:
    """
    Args:
      line:
        A line of transcript consisting of space(s) separated words.
    Returns:
      Return None.
    """
    char = line.strip().split()[0]
    syllables = pinyin(char, style=Style.TONE3, heteronym=True)
    syllables = ''.join(str(syllables[0][:]))
    for s in syllables.split(',') :
        print("{} {}".format(char, re.sub(r'[]', '', s)))


def main():
    args = get_args()
    assert Path(args.lexicon).is_file()

    with open(args.lexicon) as f:
        for line in f:
            process_line(line=line)


if __name__ == "__main__":
    main()
