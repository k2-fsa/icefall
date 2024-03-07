#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
This file prints the text field of supervisions from cutset to the console
"""

import argparse
from pathlib import Path

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        help="Input manifest",
    )
    return parser.parse_args()


def main():
    args = get_args()
    assert args.manifest.is_file(), args.manifest

    cutset = load_manifest_lazy(args.manifest)
    for c in cutset:
        for sup in c.supervisions:
            print(sup.text)


if __name__ == "__main__":
    main()
