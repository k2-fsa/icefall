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

  echo "----------lexicon.txt----------"
  head data/lexicon.txt
  echo "----"
  tail data/lexicon.txt
  echo "----"
  wc -l data/lexicon.txt
}

function train() {
  pushd ./vits
  sed -i.bak s/200/50/g ./train.py
  git diff .
  popd

  # for t in low medium high; do
  for t in low; do
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
  # for t in low medium high; do
  for t in low; do
    ./vits/export-onnx.py \
      --model-type $t \
      --epoch 1 \
      --exp-dir ./vits/exp-$t \
      --tokens data/tokens.txt \
      --speakers ./data/speakers.txt

    ls -lh vits/exp-$t/
  done
}

function test_low() {
  git clone https://huggingface.co/csukuangfj/icefall-tts-aishell3-vits-low-2024-04-06
  repo=icefall-tts-aishell3-vits-low-2024-04-06

  ./vits/export-onnx.py \
    --model-type low \
    --epoch 1000 \
    --exp-dir $repo/exp \
    --tokens $repo/data/tokens.txt \
    --speakers $repo/data/speakers.txt

  ls -lh $repo/exp/vits-epoch-1000.onnx

  python3 -m pip install sherpa-onnx

  sherpa-onnx-offline-tts \
    --vits-model=$repo/exp/vits-epoch-960.onnx \
    --vits-tokens=$repo/data/tokens.txt \
    --vits-lexicon=$repo/data/lexicon.txt \
    --num-threads=1 \
    --vits-length-scale=1.0 \
    --sid=33 \
    --output-filename=/icefall/low.wav \
    --debug=1 \
    "这是一个语音合成测试"
}


download_data
prepare_data
train
export_onnx
test_low
