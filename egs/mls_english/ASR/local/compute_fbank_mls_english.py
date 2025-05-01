#!/usr/bin/env python3
# Copyright    2023  The University of Electro-Communications
#               (Author: Teo Wen Shen)
#
# Apache-2.0

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.utils import is_module_available

# Disable PyTorch intra/inter op threading overhead
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def make_cutset_blueprints(
    mls_eng_hf_dataset_path: str = "parler-tts/mls_eng",
) -> List[Tuple[str, CutSet]]:
    if not is_module_available("datasets"):
        raise ImportError(
            "To process the MLS English HF corpus, please install datasets: pip install datasets"
        )
    from datasets import load_dataset

    dataset = load_dataset(str(mls_eng_hf_dataset_path))

    return [
        ("test",  CutSet.from_huggingface_dataset(dataset["test"],  text_key="transcript")),
        ("dev",   CutSet.from_huggingface_dataset(dataset["dev"],   text_key="transcript")),
        ("train", CutSet.from_huggingface_dataset(dataset["train"], text_key="transcript")),
    ]


def get_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-m", "--manifest-dir",
                   type=Path,
                   default=Path("data/manifests"),
                   help="Where to write JSONL cuts")
    p.add_argument("-a", "--audio-dir",
                   type=Path,
                   default=Path("data/audio"),
                   help="Where to copy raw audio")
    p.add_argument("-d", "--dl-dir",
                   type=Path,
                   required=True,
                   help="Where the HF dataset was cloned")
    p.add_argument("--fbank-dir",
                   type=Path,
                   default=Path("data/fbank"),
                   help="Where to write FBANK features")
    return p.parse_args()


def main():
    args = get_args()

    # Make sure our directories exist
    for d in (args.manifest_dir, args.audio_dir, args.fbank_dir):
        d.mkdir(parents=True, exist_ok=True)

    # If we've already computed FBANK, skip.
    done_marker = args.fbank_dir / ".mls_eng-fbank.done"
    if done_marker.exists():
        logging.info(
            "Found done-marker at %s. Skipping FBANK computation.",
            done_marker
        )
        return

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    # Prepare Lhotse cut blueprints from HF dataset
    cut_sets = make_cutset_blueprints(str(args.dl_dir))

    # Feature extractor
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    num_jobs = min(16, os.cpu_count())

    for part, cut_set in cut_sets:
        logging.info("===== Processing split: %s =====", part)

        # 1) compute & store FBANK features into fbank-dir
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            num_jobs=num_jobs,
            storage_path=(args.fbank_dir / f"mls_eng_feats_{part}").as_posix(),
            storage_type=LilcomChunkyWriter,
        )

        # 2) copy raw audio into audio-dir/<split>/
        cut_set = cut_set.save_audios(args.audio_dir / part)

        # 3) write final cuts JSONL into manifest-dir
        out_manifest = args.manifest_dir / f"mls_eng_cuts_{part}.jsonl.gz"
        cut_set.to_file(out_manifest)
        logging.info("Wrote cuts manifest to %s", out_manifest)

    # Touch the done marker so next runs skip
    done_marker.touch()
    logging.info("All FBANK computed. Done marker created at %s", done_marker)


if __name__ == "__main__":
    main()
