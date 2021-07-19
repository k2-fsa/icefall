#!/usr/bin/env python3

"""
This file downloads the librispeech dataset
to the directory data/LibriSpeech.

It's compatible with kaldi's egs/librispeech/s5/local/download_and_untar.sh .
"""


from lhotse.recipes import download_librispeech


def download_data():
    target_dir = "data"

    download_librispeech(target_dir=target_dir, dataset_parts="librispeech")


if __name__ == "__main__":
    download_data()
