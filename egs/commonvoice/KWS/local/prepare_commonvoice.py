#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Normalize Lhotse Common Voice manifests and create 16 kHz raw cuts."""

import argparse
import logging
import sys
from pathlib import Path

from lhotse import CutSet
from lhotse.recipes.utils import read_manifests_if_cached

ASR_LOCAL = Path(__file__).resolve().parents[2] / "ASR" / "local"
if str(ASR_LOCAL) not in sys.path:
    sys.path.insert(0, str(ASR_LOCAL))

from japanese_text import normalize_japanese_text


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prepare_cuts(manifest_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_parts = ("train", "dev", "test")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=manifest_dir,
        suffix="jsonl.gz",
        prefix="cv-ja",
    )
    if manifests is None or set(manifests) != set(dataset_parts):
        raise RuntimeError(
            f"Incomplete Common Voice manifests in {manifest_dir}; expected {dataset_parts}."
        )

    for partition in dataset_parts:
        output = output_dir / f"cv-ja_cuts_{partition}_raw.jsonl.gz"
        if output.is_file():
            logging.info("%s already exists; skipping", output)
            continue

        manifest = manifests[partition]
        for supervision in manifest["supervisions"]:
            supervision.text = normalize_japanese_text(str(supervision.text))

        cuts = CutSet.from_manifests(
            recordings=manifest["recordings"],
            supervisions=manifest["supervisions"],
        ).resample(16000)
        cuts = cuts.filter(lambda cut: bool(cut.supervisions[0].text))
        cuts.to_file(output)
        logging.info("Wrote normalized %s cuts to %s", partition, output)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = get_args()
    prepare_cuts(args.manifest_dir, args.output_dir)


if __name__ == "__main__":
    main()
