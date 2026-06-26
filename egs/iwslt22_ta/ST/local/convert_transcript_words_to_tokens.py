#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
"""
Convert a transcript file containing words to a corpus file containing tokens
for LM training with the help of a lexicon.

If the lexicon contains phones, the resulting LM will be a phone LM; If the
lexicon contains word pieces, the resulting LM will be a word piece LM.

If a word has multiple pronunciations, the one that appears first in the lexicon
is kept; others are removed.

If the input transcript is:

    hello zoo world hello
    world zoo
    foo zoo world hellO

and if the lexicon is

    <UNK> SPN
    hello h e l l o 2
    hello h e l l o
    world w o r l d
    zoo z o o

Then the output is

    h e l l o 2 z o o w o r l d h e l l o 2
    w o r l d z o o
    SPN z o o w o r l d SPN
"""

import argparse
from pathlib import Path
from typing import Dict, List

from generate_unique_lexicon import filter_multiple_pronunications

from icefall.lexicon import read_lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript",
        type=str,
        help="The input transcript file."
        "We assume that the transcript file consists of "
        "lines. Each line consists of space separated words.",
    )
    parser.add_argument("--lexicon", type=str, help="The input lexicon file.")
    parser.add_argument("--oov", type=str, default="<UNK>", help="The OOV word.")

    return parser.parse_args()


def process_line(lexicon: Dict[str, List[str]], line: str, oov_token: str) -> None:
    """
    Args:
      lexicon:
        A dict containing pronunciations. Its keys are words and values
        are pronunciations (i.e., tokens).
      line:
        A line of transcript consisting of space(s) separated words.
      oov_token:
        The pronunciation of the oov word if a word in `line` is not present
        in the lexicon.
    Returns:
      Return None.
    """
    s = ""
    words = line.strip().split()
    for i, w in enumerate(words):
        tokens = lexicon.get(w, oov_token)
        s += " ".join(tokens)
        s += " "
    print(s.strip())


def main():
    args = get_args()
    assert Path(args.lexicon).is_file()
    assert Path(args.transcript).is_file()
    assert len(args.oov) > 0

    # Only the first pronunciation of a word is kept
    lexicon = filter_multiple_pronunications(read_lexicon(args.lexicon))

    lexicon = dict(lexicon)

    assert args.oov in lexicon

    oov_token = lexicon[args.oov]

    with open(args.transcript) as f:
        for line in f:
            process_line(lexicon=lexicon, line=line, oov_token=oov_token)


if __name__ == "__main__":
    main()
