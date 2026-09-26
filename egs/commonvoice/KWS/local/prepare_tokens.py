#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Create a phone token table from normalized Common Voice training text."""

import argparse
from collections import Counter
from pathlib import Path

from lhotse import CutSet

from japanese_phones import JAPANESE_PHONE_INVENTORY, text_to_phones


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", type=Path, required=True)
    parser.add_argument("--lang-dir", type=Path, required=True)
    return parser.parse_args()


def write_tokens(cuts: CutSet, lang_dir: Path) -> Counter:
    counts: Counter = Counter()
    # Retain zero-count standard phones so dev/test or a user keyword can be
    # represented even when its phone is absent from this training split.
    counts.update({phone: 0 for phone in JAPANESE_PHONE_INVENTORY})
    for cut in cuts:
        counts.update(text_to_phones(cut.supervisions[0].text))

    if not counts:
        raise RuntimeError("No phones were found in the training cuts")

    lang_dir.mkdir(parents=True, exist_ok=True)
    tokens = ["<blk>", "<unk>", *sorted(counts)]
    (lang_dir / "tokens.txt").write_text(
        "".join(f"{token} {index}\n" for index, token in enumerate(tokens)),
        encoding="utf-8",
    )
    (lang_dir / "phone_counts.tsv").write_text(
        "".join(f"{phone}\t{counts[phone]}\n" for phone in sorted(counts)),
        encoding="utf-8",
    )
    return counts


def main() -> None:
    args = get_args()
    counts = write_tokens(CutSet.from_file(args.cuts), args.lang_dir)
    print(f"Wrote {len(counts)} lexical phones to {args.lang_dir / 'tokens.txt'}")


if __name__ == "__main__":
    main()
