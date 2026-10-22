#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compute persistent 80-bin fbank features for Japanese Common Voice cuts."""

import argparse
import gzip
import json
import logging
import os
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    KaldifeatFbank,
    KaldifeatFbankConfig,
    LilcomChunkyWriter,
    set_audio_duration_mismatch_tolerance,
    set_caching_enabled,
)


def write_cuts_utf8(cuts: CutSet, output_cuts: Path) -> None:
    """Write a completed feature manifest atomically with Japanese text intact.

    Lhotse 1.28.0's JSONL writer can select an ASCII encoding for gzip files
    even when Python's preferred encoding is UTF-8.  Explicitly opening the
    stream also lets the final manifest remain an unambiguous success marker:
    an interrupted write never looks like a completed partition.
    """
    temporary = output_cuts.with_name(f".{output_cuts.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            for cut in cuts:
                stream.write(json.dumps(cut.to_dict(), ensure_ascii=False))
                stream.write("\n")
        os.replace(temporary, output_cuts)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-duration", type=float, default=200.0)
    return parser.parse_args()


def compute_features(args: argparse.Namespace) -> None:
    fbank_dir = args.data_dir / "fbank"
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    set_audio_duration_mismatch_tolerance(0.05)
    set_caching_enabled(False)
    logging.info("Feature extraction device: %s", device)

    for partition in ("train", "dev", "test"):
        raw_cuts = fbank_dir / f"cv-ja_cuts_{partition}_raw.jsonl.gz"
        output_cuts = fbank_dir / f"cv-ja_cuts_{partition}.jsonl.gz"
        completion_marker = output_cuts.with_name(f".{output_cuts.name}.done")
        if output_cuts.is_file() and completion_marker.is_file():
            logging.info("%s already exists; skipping", output_cuts)
            continue
        if not raw_cuts.is_file():
            raise FileNotFoundError(f"Missing raw cuts: {raw_cuts}")

        cuts = CutSet.from_file(raw_cuts)
        cuts = cuts.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=str(fbank_dir / f"cv-ja_feats_{partition}"),
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            storage_type=LilcomChunkyWriter,
            # The completed cut manifest is the atomic success marker. If a
            # prior attempt stops before that marker is written, recompute the
            # corresponding feature store instead of trusting partial chunks.
            overwrite=True,
        )
        write_cuts_utf8(cuts, output_cuts)
        completion_marker.write_text("completed\n", encoding="utf-8")
        logging.info("Wrote fbank cuts to %s", output_cuts)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    compute_features(get_args())


if __name__ == "__main__":
    main()
