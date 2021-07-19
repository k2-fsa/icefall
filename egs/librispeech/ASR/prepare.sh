#!/usr/bin/env bash

set -eou pipefail

stage=-1
stop_stage=100

. local/parse_options.sh || exit 1

mkdir -p data

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  echo "stage -1: Download LM"
  mkdir -p data/lm
  ./local/download_lm.py
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "stage 0: Download data"

  # If you have pre-downloaded it to /path/to/LibriSpeech,
  # you can create a symlink to avoid downloading it again:
  #
  #   ln -sfv /path/to/LibriSpeech data/
  #

  mkdir -p data/LibriSpeech

  if [ ! -f data/LibriSpeech/train-other-500/.completed ]; then
    # It's compatible with kaldi's egs/librispeech/s5/local/download_and_untar.sh
    ./local/download_data.py
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink to avoid downloading it again:
  #
  #   ln -s /path/to/musan data/
  #
  if [ ! -e data/musan ]; then
    wget https://www.openslr.org/resources/17/musan.tar.gz
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Stage 1: Prepare librispeech manifest"
  # We assume that you have downloaded the librispeech corpus
  # to data/LibriSpeech
  mkdir -p data/manifests
  ./local/prepare_librispeech_manifest.py
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  ./local/prepare_musan_manifest.py
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Stage 3: Compute fbank for librispeech"
  mkdir -p data/fbank
  ./local/compute_fbank_librispeech.py
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Stage 4: Compute fbank for librispeech"
  mkdir -p data/fbank
  ./local/compute_fbank_musan.py
fi
