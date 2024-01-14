#!/usr/bin/env bash

# This script computes fbank features for the test-clean and test-other datasets.
# The computed features are saved to ~/tmp/fbank-libri and are
# cached for later runs

set -e

export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH

mkdir ~/tmp/fbank-libri
cd egs/librispeech/ASR
mkdir -p data
cd data
[ ! -e fbank ] && ln -s ~/tmp/fbank-libri fbank
cd ..
./local/compute_fbank_librispeech.py --dataset 'test-clean test-other'
ls -lh data/fbank/
