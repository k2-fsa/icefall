#!/usr/bin/env python3

import argparse
from pathlib import Path

from icefall.lexicon import read_lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        Generated files by this script are saved into this directory.
        """,
    )

    parser.add_argument(
        "--otc-token",
        type=str,
        help="OTC token to be added to words.txt",
    )

    return parser.parse_args()


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    otc_token = args.otc_token

    lexicon = read_lexicon(lang_dir / "lexicon.txt")
    ans = set()
    for word, _ in lexicon:
        ans.add(word)
    sorted_ans = sorted(list(ans))
    words = ["<eps>"] + sorted_ans + [otc_token] + ["#0", "<s>", "</s>"]

    words_file = lang_dir / "words.txt"
    with open(words_file, "w") as wf:
        for i, word in enumerate(words):
            wf.write(f"{word} {i}\n")


if __name__ == "__main__":
    main()
