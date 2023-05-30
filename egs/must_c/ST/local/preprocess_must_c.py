#!/usr/bin/env python3
import argparse
import logging
import re
from pathlib import Path
from functools import partial

from normalize_punctuation import normalize_punctuation
from lhotse.recipes.utils import read_manifests_if_cached


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Manifest directory",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        required=True,
        help="Target language, e.g., zh, de, fr.",
    )
    return parser.parse_args()


def preprocess_must_c(manifest_dir: Path, tgt_lang: str):
    print(manifest_dir)

    normalize_punctuation_lang = partial(normalize_punctuation, lang=tgt_lang)

    prefix = "must_c"
    suffix = "jsonl.gz"
    parts = ["dev"]
    for p in parts:
        name = f"en-{tgt_lang}_{p}"

        manifests = read_manifests_if_cached(
            dataset_parts=name,
            output_dir=manifest_dir,
            prefix=prefix,
            suffix=suffix,
            types=("supervisions",),
        )
        if name not in manifests:
            raise RuntimeError(f"Processing {p} failed.")

        supervisions = manifests[name]["supervisions"]
        if True:
            supervisions2 = supervisions.transform_text(normalize_punctuation_lang)

        for s, s2 in zip(supervisions, supervisions2):
            if s.text != s2.text:
                print(s.text)
                print(s2.text)
                print("-" * 10)


def main():
    args = get_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    logging.info(vars(args))
    assert args.manifest_dir.is_dir(), args.manifest_dir

    preprocess_must_c(
        manifest_dir=args.manifest_dir,
        tgt_lang=args.tgt_lang,
    )


if __name__ == "__main__":
    main()
