#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)
"""
This script normalizes transcripts from supervisions.

Usage:
  ./local/preprocess_must_c.py \
    --manifest-dir ./data/manifests/v1.0/ \
    --tgt-lang de
"""

import argparse
import logging
import re
from functools import partial
from pathlib import Path

from lhotse.recipes.utils import read_manifests_if_cached
from normalize_punctuation import normalize_punctuation
from remove_non_native_characters import remove_non_native_characters
from remove_punctuation import remove_punctuation


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
    normalize_punctuation_lang = partial(normalize_punctuation, lang=tgt_lang)
    remove_non_native_characters_lang = partial(
        remove_non_native_characters, lang=tgt_lang
    )

    prefix = "must_c"
    suffix = "jsonl.gz"
    parts = ["dev", "tst-COMMON", "tst-HE", "train"]
    for p in parts:
        logging.info(f"Processing {p}")
        name = f"en-{tgt_lang}_{p}"

        # norm: normalization
        # rm: remove punctuation
        dst_name = manifest_dir / f"must_c_supervisions_{name}_norm_rm.jsonl.gz"
        if dst_name.is_file():
            logging.info(f"{dst_name} exists - skipping")
            continue

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
        supervisions = supervisions.transform_text(normalize_punctuation_lang)
        supervisions = supervisions.transform_text(remove_punctuation)
        supervisions = supervisions.transform_text(lambda x: x.lower())
        supervisions = supervisions.transform_text(remove_non_native_characters_lang)
        supervisions = supervisions.transform_text(lambda x: re.sub(" +", " ", x))

        supervisions.to_file(dst_name)


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
