#!/usr/bin/env python3

"""
This file generates manifests for the librispeech dataset.
It expects the dataset is saved in data/LibriSpeech
and the generated manifests are saved in data/manifests.
"""

import os
from pathlib import Path

from lhotse.recipes import prepare_librispeech


def prepare_librispeech_mainfest():
    corpus_dir = Path("data/LibriSpeech")
    output_dir = Path("data/manifests")
    num_jobs = min(15, os.cpu_count())

    librispeech_manifests = prepare_librispeech(
        corpus_dir=corpus_dir,
        dataset_parts="auto",
        output_dir=output_dir,
        num_jobs=num_jobs,
    )


if __name__ == "__main__":
    prepare_librispeech_mainfest()
