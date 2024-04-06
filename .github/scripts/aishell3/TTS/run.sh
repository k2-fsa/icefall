#!/usr/bin/env bash

set -ex

python3 -m pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
python3 -m pip install numba
python3 -m pip install pypinyin
python3 -m pip install cython

apt-get update
apt-get install -y jq

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/aishell3/TTS

sed -i.bak s/1000/10/g ./prepare.sh


function download_data() {
  mkdir download
  pushd download
  curl -SL -O https://huggingface.co/csukuangfj/aishell3-ci-data/resolve/main/aishell3.tar.bz2
  tar xf aishell3.tar.bz2
  rm aishell3.tar.bz2
  ls -lh
  popd
}

function prepare_data() {
  ./prepare.sh

  echo "----------tokens.txt----------"
  cat data/tokens.txt
  echo "------------------------------"
  wc -l data/tokens.txt
  echo "------------------------------"
}

function train() {
  pushd ./vits
  sed -i.bak s/200/50/g ./train.py
  git diff .
  popd

  for t in low medium high; do
    ./vits/train.py \
      --exp-dir vits/exp-$t \
      --model-type $t \
      --num-epochs 1 \
      --save-every-n 1 \
      --num-buckets 2 \
      --tokens data/tokens.txt \
      --max-duration 20

    ls -lh vits/exp-$t
  done
}

function export_onnx() {
  for t in low medium high; do
    ./vits/export-onnx.py \
      --model-type $t \
      --epoch 1 \
      --exp-dir ./vits/exp-$t \
      --tokens data/tokens.txt
      --speakers ./data/speakers.txt

    ls -lh vits/exp-$t/
  done
}

function test_low() {
  echo "TODO"
}


download_data
prepare_data
train
export_onnx
test_low
