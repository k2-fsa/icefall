#!/usr/bin/env bash

set -ex

apt-get update
apt-get install -y sox

python3 -m pip install numba conformer==0.3.2 diffusers librosa
python3 -m pip install jieba


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/baker_zh/TTS

sed -i.bak s/600/8/g ./prepare.sh
sed -i.bak s/"first 100"/"first 3"/g ./prepare.sh
sed -i.bak s/500/5/g ./prepare.sh
git diff

function prepare_data() {
  # We have created a subset of the data for testing
  #
  mkdir -p download
  pushd download
  wget -q https://huggingface.co/csukuangfj/tmp-files/resolve/main/BZNSYP-samples.tar.bz2
  tar xvf BZNSYP-samples.tar.bz2
  mv BZNSYP-samples BZNSYP
  rm BZNSYP-samples.tar.bz2
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
  curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v2

  ./matcha/infer.py \
    --num-buckets 2 \
    --epoch 1 \
    --exp-dir ./matcha/exp \
    --tokens data/tokens.txt \
    --cmvn ./data/fbank/cmvn.json \
    --vocoder ./generator_v2 \
    --input-text "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。" \
    --output-wav ./generated.wav

  ls -lh *.wav
  soxi ./generated.wav
  rm -v ./generated.wav
  rm -v generator_v2
}

function export_onnx() {
  pushd matcha/exp
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-baker-matcha-zh-2024-12-27/resolve/main/epoch-2000.pt
  popd

  pushd data/fbank
  rm -v *.json
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-baker-matcha-zh-2024-12-27/resolve/main/cmvn.json
  popd

  ./matcha/export_onnx.py \
    --exp-dir ./matcha/exp \
    --epoch 2000 \
    --tokens ./data/tokens.txt \
    --cmvn ./data/fbank/cmvn.json

  ls -lh *.onnx

  if false; then
    # The CI machine does not have enough memory to run it
    #
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v1
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v2
    curl -SL -O https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v3
    python3 ./matcha/export_onnx_hifigan.py
  else
    curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/exp/hifigan_v1.onnx
    curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/exp/hifigan_v2.onnx
    curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28/resolve/main/exp/hifigan_v3.onnx
  fi

  ls -lh *.onnx

  python3 ./matcha/generate_lexicon.py

  for v in v1 v2 v3; do
    python3 ./matcha/onnx_pretrained.py \
     --acoustic-model ./model-steps-6.onnx \
     --vocoder ./hifigan_$v.onnx \
     --tokens ./data/tokens.txt \
     --lexicon ./lexicon.txt \
     --input-text "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。" \
     --output-wav /icefall/generated-matcha-tts-steps-6-$v.wav
  done

  ls -lh /icefall/*.wav
  soxi /icefall/generated-matcha-tts-steps-6-*.wav
  cp ./model-steps-*.onnx /icefall

  d=matcha-icefall-zh-baker
  mkdir $d
  cp -v data/tokens.txt $d
  cp -v lexicon.txt $d
  cp model-steps-3.onnx $d
  pushd $d
  curl -SL -O https://github.com/csukuangfj/cppjieba/releases/download/sherpa-onnx-2024-04-19/dict.tar.bz2
  tar xvf dict.tar.bz2
  rm dict.tar.bz2

  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-aishell3-vits-low-2024-04-06/resolve/main/data/date.fst
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-aishell3-vits-low-2024-04-06/resolve/main/data/number.fst
  curl -SL -O https://huggingface.co/csukuangfj/icefall-tts-aishell3-vits-low-2024-04-06/resolve/main/data/phone.fst

cat >README.md <<EOF
# Introduction

This model is trained using the dataset from
https://en.data-baker.com/datasets/freeDatasets/

The dataset contains 10000 Chinese sentences of a native Chinese female speaker,
which is about 12 hours.

**Note**: The dataset is for non-commercial use only.

You can find the training code at
https://github.com/k2-fsa/icefall/tree/master/egs/baker_zh/TTS
EOF

  ls -lh
  popd
  tar cvjf $d.tar.bz2 $d
  mv $d.tar.bz2 /icefall
  mv $d /icefall
}

prepare_data
train
infer
export_onnx

rm -rfv generator_v* matcha/exp
git checkout .
