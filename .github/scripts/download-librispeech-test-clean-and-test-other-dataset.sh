#!/usr/bin/env bash

# This script downloads the test-clean and test-other datasets
# of LibriSpeech and unzip them to the folder ~/tmp/download,
# which is cached by GitHub actions for later runs.
#
# You will find directories ~/tmp/download/LibriSpeech after running
# this script.

set -e

mkdir ~/tmp/download
cd egs/librispeech/ASR
ln -s ~/tmp/download .
cd download
wget -q --no-check-certificate https://www.openslr.org/resources/12/test-clean.tar.gz
tar xf test-clean.tar.gz
rm test-clean.tar.gz

wget -q --no-check-certificate https://www.openslr.org/resources/12/test-other.tar.gz
tar xf test-other.tar.gz
rm test-other.tar.gz
pwd
ls -lh
ls -lh LibriSpeech
