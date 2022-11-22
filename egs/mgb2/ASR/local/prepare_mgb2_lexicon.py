#!/usr/bin/env python3

# Copyright      2022  Amir Hussein
# Apache 2.0

# This script prepares givel a column of words lexicon.

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="""Creates the list of characters and words in lexicon"""
    )
    parser.add_argument("input", type=str, help="""Input list of words file""")
    parser.add_argument("output", type=str, help="""output graphemic lexicon""")
    args = parser.parse_args()
    return args


def main():
    lex = {}
    args = get_args()
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            characters = list(line)
            characters = " ".join(["V" if char == "*" else char for char in characters])
            lex[line] = characters

    with open(args.output, "w", encoding="utf-8") as fp:
        for key in sorted(lex):
            fp.write(key + "  " + lex[key] + "\n")


if __name__ == "__main__":
    main()
