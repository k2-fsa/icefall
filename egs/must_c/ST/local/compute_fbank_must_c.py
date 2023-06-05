#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This file computes fbank features of the MuST-C dataset.
It looks for manifests in the directory "in_dir" and write
generated features to "out_dir".
"""
import argparse
import logging
from pathlib import Path

import torch
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    FeatureSet,
    LilcomChunkyWriter,
    load_manifest,
)

from icefall.utils import str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Input manifest directory",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory where generated fbank features are saved.",
    )

    parser.add_argument(
        "--tgt-lang",
        type=str,
        required=True,
        help="Target language, e.g., zh, de, fr.",
    )

    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Number of jobs for computing features",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=False,
        help="""True to enable speed perturb with factors 0.9 and 1.1 on
        the train subset. False (by default) to disable speed perturb.
        """,
    )

    return parser.parse_args()


def compute_fbank_must_c(
    in_dir: Path,
    out_dir: Path,
    tgt_lang: str,
    num_jobs: int,
    perturb_speed: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = Fbank(FbankConfig(num_mel_bins=80))

    parts = ["dev", "tst-COMMON", "tst-HE", "train"]

    prefix = "must_c"
    suffix = "jsonl.gz"
    for p in parts:
        logging.info(f"Processing {p}")

        cuts_path = f"{out_dir}/{prefix}_feats_en-{tgt_lang}_{p}"
        if perturb_speed and p == "train":
            cuts_path += "_sp"

        cuts_path += ".jsonl.gz"

        if Path(cuts_path).is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        recordings_filename = in_dir / f"{prefix}_recordings_en-{tgt_lang}_{p}.jsonl.gz"
        supervisions_filename = (
            in_dir / f"{prefix}_supervisions_en-{tgt_lang}_{p}_norm_rm.jsonl.gz"
        )
        assert recordings_filename.is_file(), recordings_filename
        assert supervisions_filename.is_file(), supervisions_filename
        cut_set = CutSet.from_manifests(
            recordings=load_manifest(recordings_filename),
            supervisions=load_manifest(supervisions_filename),
        )
        if perturb_speed and p == "train":
            logging.info("Speed perturbing for the train dataset")
            cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
            storage_path = f"{out_dir}/{prefix}_feats_en-{tgt_lang}_{p}_sp"
        else:
            storage_path = f"{out_dir}/{prefix}_feats_en-{tgt_lang}_{p}"

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=storage_path,
            num_jobs=num_jobs,
            storage_type=LilcomChunkyWriter,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    args = get_args()
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    logging.info(vars(args))
    assert args.in_dir.is_dir(), args.in_dir

    compute_fbank_must_c(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        tgt_lang=args.tgt_lang,
        num_jobs=args.num_jobs,
        perturb_speed=args.perturb_speed,
    )


if __name__ == "__main__":
    main()
