#!/usr/bin/env bash

# We use the model from
# https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main
# as an example

export CUDA_VISIBLE_DEVICES=

if [ ! -f ./icefall-librispeech-rnn-lm/exp/pretrained.pt ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
  pushd icefall-librispeech-rnn-lm/exp
  git lfs pull --include "pretrained.pt"
  ln -s pretrained.pt epoch-99.pt
  popd
fi

python3 ./export.py \
  --exp-dir ./icefall-librispeech-rnn-lm/exp \
  --epoch 99 \
  --avg 1 \
  --vocab-size 500 \
  --embedding-dim 2048 \
  --hidden-dim 2048 \
  --num-layers 3 \
  --tie-weights 1 \
  --jit 1

