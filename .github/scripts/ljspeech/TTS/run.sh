#!/usr/bin/env bash

set -ex

python3 -m pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
python3 -m pip install espnet_tts_frontend
python3 -m pip install numba

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/ljspeech/TTS

sed -i.bak s/600/8/g ./prepare.sh
sed -i.bak s/"first 100"/"first 3"/g ./prepare.sh
sed -i.bak s/500/5/g ./prepare.sh
git diff

function prepare_data() {
  # We have created a subset of the data for testing
  #
  mkdir -p download
  pushd download
  wget -q https://huggingface.co/csukuangfj/ljspeech-subset-for-ci-test/resolve/main/LJSpeech-1.1.tar.bz2
  tar xvf LJSpeech-1.1.tar.bz2
  popd

  ./prepare.sh
  tree .
}

function train() {
  pushd ./vits
  sed -i.bak s/200/3/g ./train.py
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

function infer() {
  for t in low medium high; do
    ./vits/infer.py \
      --num-buckets 2 \
      --model-type $t \
      --epoch 1 \
      --exp-dir ./vits/exp-$t \
      --tokens data/tokens.txt \
      --max-duration 20
  done
}

function export_onnx() {
  for t in low medium high; do
    ./vits/export-onnx.py \
      --model-type $t \
      --epoch 1 \
      --exp-dir ./vits/exp-$t \
      --tokens data/tokens.txt

    ls -lh vits/exp-$t/
  done
}

function test_medium() {
  git clone https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-medium-2024-03-12

  ./vits/export-onnx.py \
    --model-type medium \
    --epoch 820 \
    --exp-dir ./icefall-tts-ljspeech-vits-medium-2024-03-12/exp \
    --tokens ./icefall-tts-ljspeech-vits-medium-2024-03-12/data/tokens.txt

  ls -lh ./icefall-tts-ljspeech-vits-medium-2024-03-12/exp

  ./vits/test_onnx.py \
    --model-filename ./icefall-tts-ljspeech-vits-medium-2024-03-12/exp/vits-epoch-820.onnx \
    --tokens ./icefall-tts-ljspeech-vits-medium-2024-03-12/data/tokens.txt \
    --output-filename /icefall/test-medium.wav

  ls -lh /icefall/test-medium.wav

  d=/icefall/vits-icefall-en_US-ljspeech-medium
  mkdir $d
  cp -v ./icefall-tts-ljspeech-vits-medium-2024-03-12/data/tokens.txt $d/
  cp -v ./icefall-tts-ljspeech-vits-medium-2024-03-12/exp/vits-epoch-820.onnx $d/model.onnx

  rm -rf icefall-tts-ljspeech-vits-medium-2024-03-12

  pushd $d
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
  tar xf espeak-ng-data.tar.bz2
  rm espeak-ng-data.tar.bz2
  cd ..
  tar cjf vits-icefall-en_US-ljspeech-medium.tar.bz2 vits-icefall-en_US-ljspeech-medium
  rm -rf vits-icefall-en_US-ljspeech-medium
  ls -lh *.tar.bz2
  popd
}

function test_low() {
  git clone https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-low-2024-03-12

  ./vits/export-onnx.py \
    --model-type low \
    --epoch 1600 \
    --exp-dir ./icefall-tts-ljspeech-vits-low-2024-03-12/exp \
    --tokens ./icefall-tts-ljspeech-vits-low-2024-03-12/data/tokens.txt

  ls -lh ./icefall-tts-ljspeech-vits-low-2024-03-12/exp

  ./vits/test_onnx.py \
    --model-filename ./icefall-tts-ljspeech-vits-low-2024-03-12/exp/vits-epoch-1600.onnx \
    --tokens ./icefall-tts-ljspeech-vits-low-2024-03-12/data/tokens.txt \
    --output-filename /icefall/test-low.wav

  ls -lh /icefall/test-low.wav

  d=/icefall/vits-icefall-en_US-ljspeech-low
  mkdir $d
  cp -v ./icefall-tts-ljspeech-vits-low-2024-03-12/data/tokens.txt $d/
  cp -v ./icefall-tts-ljspeech-vits-low-2024-03-12/exp/vits-epoch-1600.onnx $d/model.onnx

  rm -rf icefall-tts-ljspeech-vits-low-2024-03-12

  pushd $d
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
  tar xf espeak-ng-data.tar.bz2
  rm espeak-ng-data.tar.bz2
  cd ..
  tar cjf vits-icefall-en_US-ljspeech-low.tar.bz2 vits-icefall-en_US-ljspeech-low
  rm -rf vits-icefall-en_US-ljspeech-low
  ls -lh *.tar.bz2
  popd
}

prepare_data
train
infer
export_onnx
rm -rf vits/exp-{low,medium,high}
test_medium
test_low
