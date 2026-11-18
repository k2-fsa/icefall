#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Convert Japanese written keywords to sherpa-onnx phone keyword lines."""

import argparse
from pathlib import Path

from japanese_phones import text_to_phones
from prepare_kws_eval import read_keywords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords-file", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = {
        line.split()[0] for line in args.tokens.read_text(encoding="utf-8").splitlines()
    }
    lines = []
    for keyword in read_keywords(args.keywords_file):
        phones = text_to_phones(keyword)
        if not phones or not set(phones) <= tokens:
            raise ValueError(f"Missing phone tokens for {keyword!r}")
        lines.append(" ".join(phones) + " @" + keyword)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} sherpa-onnx keywords to {args.output}")


if __name__ == "__main__":
    main()
