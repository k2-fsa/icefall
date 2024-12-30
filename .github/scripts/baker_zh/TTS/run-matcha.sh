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

prepare_data
