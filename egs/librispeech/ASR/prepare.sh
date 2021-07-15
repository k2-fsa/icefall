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

  # If you have pre-downloaded it in /path/to/LibriSpeech
  # Just run: ln -sfv /path/to/LibriSpeech data/
  mkdir -p data/LibriSpeech
  # TODO
fi
