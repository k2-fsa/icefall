#!/usr/bin/env bash

# This script downloads the pre-computed fbank features for
# dev and test datasets of GigaSpeech.
#
# You will find directories `~/tmp/giga-dev-dataset-fbank` after running
# this script.

set -e

mkdir -p ~/tmp
cd ~/tmp

git lfs install
git clone https://huggingface.co/csukuangfj/giga-dev-dataset-fbank

ls -lh giga-dev-dataset-fbank/data/fbank
