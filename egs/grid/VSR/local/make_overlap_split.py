#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0

"""Build the GRID overlapped-speaker split (LipNet protocol).

Pools the unseen-speaker train and test cuts, then holds out 255 utterances
per speaker (shuffled with seed 42) as the test set and keeps the rest for
training. The cuts are copied unchanged, so the new manifests point at the
same feature shards as the source ones; no features are recomputed.

  python local/make_overlap_split.py \
      [--src data/avhubert_mf_l8] [--dst data/avhubert_mf_l8_overlap]
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

HELD_OUT = 255

cli = argparse.ArgumentParser()
cli.add_argument("--src", type=Path, default=Path("data/avhubert_mf_l8"))
cli.add_argument("--dst", type=Path, default=Path("data/avhubert_mf_l8_overlap"))
opts = cli.parse_args()
rng = random.Random(42)

by_spk = defaultdict(list)
for name in ("grid_cuts_train.jsonl.gz", "grid_cuts_test.jsonl.gz"):
    with gzip.open(opts.src / name, "rt") as f:
        for line in f:
            by_spk[json.loads(line)["id"].split("_")[0]].append(line)

opts.dst.mkdir(parents=True, exist_ok=True)
n_tr = n_te = 0
with gzip.open(opts.dst / "grid_cuts_train.jsonl.gz", "wt") as ftr, gzip.open(
    opts.dst / "grid_cuts_test.jsonl.gz", "wt"
) as fte:
    for spk in sorted(by_spk, key=lambda s: int(s[1:])):
        lines = by_spk[spk]
        rng.shuffle(lines)
        for line in lines[:HELD_OUT]:
            fte.write(line)
            n_te += 1
        for line in lines[HELD_OUT:]:
            ftr.write(line)
            n_tr += 1
print(
    f"speakers {len(by_spk)}; train {n_tr}; test {n_te} "
    f"({HELD_OUT} per speaker) -> {opts.dst}"
)
