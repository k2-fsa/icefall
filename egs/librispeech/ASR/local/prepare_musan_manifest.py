#!/usr/bin/env python3

"""
This file generates manifests for the musan dataset.
It expects the dataset is saved in data/musan
and the generated manifests are saved in data/manifests.
"""

from pathlib import Path

from lhotse.recipes import prepare_musan


def prepare_musan_mainfest():
    corpus_dir = Path("data/musan")
    output_dir = Path("data/manifests")

    prepare_musan(corpus_dir=corpus_dir, output_dir=output_dir)


if __name__ == "__main__":
    prepare_musan_mainfest()
