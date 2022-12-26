#!/usr/bin/env bash

# This script assumes that test-clean and test-other are downloaded
# to egs/librispeech/ASR/download/LibriSpeech and generates manifest
# files in egs/librispeech/ASR/data/manifests

set -e

cd egs/librispeech/ASR
[ ! -e download ] && ln -s ~/tmp/download .
mkdir -p data/manifests
lhotse prepare librispeech -j 2 -p test-clean -p test-other ./download/LibriSpeech data/manifests
ls -lh data/manifests
