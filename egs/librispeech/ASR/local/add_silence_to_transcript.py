#!/usr/bin/env python3

# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
'''
Add silence with a given probability after each word in the transcript.

If the input transcript contains:

    hello world
    foo bar koo
    zoo

Then the output transcript **may** look like the following:

    !SIL hello !SIL world !SIL
    foo bar !SIL koo !SIL
    !SIL zoo !SIL

(Assume !SIL represents silence.)
'''

from pathlib import Path

import argparse
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript',
                        type=str,
                        help='The input transcript file.'
                        'We assume that the transcript file consists of '
                        'lines. Each line consists of space separated words.')
    parser.add_argument('--sil-word',
                        type=str,
                        default='!SIL',
                        help='The word that represents silence.')
    parser.add_argument('--sil-prob',
                        type=float,
                        default=0.5,
                        help='The probability for adding a '
                        'silence after each world.')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help='The seed for random number generators.')

    return parser.parse_args()


def need_silence(sil_prob: float) -> bool:
    '''
    Args:
      sil_prob:
        The probability to add a silence.
    Returns:
      Return True if a silence is needed.
      Return False otherwise.
    '''
    return random.uniform(0, 1) <= sil_prob


def process_line(line: str, sil_word: str, sil_prob: float) -> None:
    '''Process a single line from the transcript.

    Args:
      line:
        A str containing space separated words.
      sil_word:
        The symbol indicating silence.
      sil_prob:
        The probability for adding a silence after each word.
    Returns:
      Return None.
    '''
    words = line.strip().split()
    for i, word in enumerate(words):
        if i == 0:
            # beginning of the line
            if need_silence(sil_prob):
                print(sil_word, end=' ')

        print(word, end=' ')

        if need_silence(sil_prob):
            print(sil_word, end=' ')

        # end of the line, print a new line
        if i == len(words) - 1:
            print()


def main():
    args = get_args()
    random.seed(args.seed)

    assert Path(args.transcript).is_file()
    assert len(args.sil_word) > 0
    assert 0 < args.sil_prob < 1

    with open(args.transcript) as f:
        for line in f:
            process_line(line=line,
                         sil_word=args.sil_word,
                         sil_prob=args.sil_prob)


if __name__ == '__main__':
    main()
