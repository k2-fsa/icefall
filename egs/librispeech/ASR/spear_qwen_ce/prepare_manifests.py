#!/usr/bin/env python3
"""Prepare LibriSpeech and MUSAN cuts."""

import argparse
import logging
import random
from pathlib import Path

from lhotse import CutSet, combine
from lhotse.recipes.librispeech import prepare_librispeech
from lhotse.recipes.musan import prepare_musan

TRAIN_SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")
SPLITS = (*TRAIN_SPLITS, "dev-clean", "dev-other", "test-clean", "test-other")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--musan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/spear_fbank"))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--num-jobs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = prepare_librispeech(
        corpus_dir=args.audio_root.resolve(),
        dataset_parts=args.splits,
        num_jobs=args.num_jobs,
    )
    training_cuts = {}
    for split, manifest in manifests.items():
        cuts = CutSet.from_manifests(
            recordings=manifest["recordings"],
            supervisions=manifest["supervisions"],
        )
        if split in TRAIN_SPLITS:
            cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
            training_cuts[split] = cuts
        cuts.to_file(args.output_dir / f"librispeech_cuts_{split}.jsonl.gz")

    if all(split in training_cuts for split in TRAIN_SPLITS):
        cuts = combine(training_cuts[split] for split in TRAIN_SPLITS).to_eager()
        cuts.shuffle(rng=random.Random(args.seed)).to_file(
            args.output_dir / "librispeech_cuts_train-all-shuf.jsonl.gz"
        )

    manifests = prepare_musan(args.musan_root.resolve())
    musan_cuts = (
        CutSet.from_manifests(
            recordings=combine(part["recordings"] for part in manifests.values())
        )
        .cut_into_windows(10.0)
        .filter(lambda cut: cut.duration > 5.0)
    )
    musan_cuts.to_file(args.output_dir / "musan_cuts.jsonl.gz")
    logging.info("Prepared cuts in %s", args.output_dir)


if __name__ == "__main__":
    main()
