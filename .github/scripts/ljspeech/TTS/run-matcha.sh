#!/usr/bin/env bash

set -ex

apt-get update
apt-get install -y sox

python3 -m pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
python3 -m pip install espnet_tts_frontend
python3 -m pip install numba conformer==0.3.2 diffusers librosa

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
  pushd ./matcha
  sed -i.bak s/1500/3/g ./train.py
  git diff .
  popd

  ./matcha/train.py \
    --exp-dir matcha/exp \
    --num-epochs 1 \
    --save-every-n 1 \
    --num-buckets 2 \
    --tokens data/tokens.txt \
    --max-duration 20

    ls -lh matcha/exp
}

function infer() {

  curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v1

  ./matcha/inference.py \
    --epoch 1 \
    --exp-dir ./matcha/exp \
    --tokens data/tokens.txt \
    --vocoder ./generator_v1 \
    --input-text "how are you doing?" \
    --output-wav ./generated.wav

  ls -lh *.wav
  soxi ./generated.wav
  rm -v ./generated.wav
  rm -v generator_v1
}

function export_onnx() {
  pushd matcha/exp
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/exp/epoch-4000.pt
  popd

  pushd data/fbank
  rm -v *.json
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/data/cmvn.json
  popd

  ./matcha/export_onnx.py \
    --exp-dir ./matcha/exp \
    --epoch 4000 \
    --tokens ./data/tokens.txt \
    --cmvn ./data/fbank/cmvn.json

  ls -lh *.onnx

  if false; then
    # THe CI machine does not have enough memory to run it
    #
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v1
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v2
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v3
    python3 ./matcha/export_onnx_hifigan.py
  else
    curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/exp/hifigan_v1.onnx
  fi

  ls -lh *.onnx

  python3 ./matcha/onnx_pretrained.py \
   --acoustic-model ./model-steps-6.onnx \
   --vocoder ./hifigan_v1.onnx \
   --tokens ./data/tokens.txt \
   --input-text "how are you doing?" \
   --output-wav /icefall/generated-matcha-tts-steps-6-v1.wav

  ls -lh /icefall/*.wav
  soxi /icefall/generated-matcha-tts-steps-6-v1.wav
}

prepare_data
train
infer
export_onnx

rm -rfv generator_v* matcha/exp
