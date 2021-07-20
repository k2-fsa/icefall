#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
"""
This file downloads librispeech LM files to data/lm
"""

import gzip
import os
import shutil
from pathlib import Path

from lhotse.utils import urlretrieve_progress
from tqdm.auto import tqdm


def download_lm():
    url = "http://www.openslr.org/resources/11"
    target_dir = Path("data/lm")

    files_to_download = (
        "3-gram.pruned.1e-7.arpa.gz",
        "4-gram.arpa.gz",
        "librispeech-vocab.txt",
        "librispeech-lexicon.txt",
    )

    for f in tqdm(files_to_download, desc="Downloading LibriSpeech LM files"):
        filename = target_dir / f
        if filename.is_file() is False:
            urlretrieve_progress(
                f"{url}/{f}", filename=filename, desc=f"Downloading {filename}",
            )
        else:
            print(f'{filename} already exists - skipping')

        if ".gz" in str(filename):
            unzip_file = Path(os.path.splitext(filename)[0])
            if unzip_file.is_file() is False:
                with gzip.open(filename, "rb") as f_in:
                    with open(unzip_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print(f'{unzip_file} already exist - skipping')


if __name__ == "__main__":
    download_lm()
