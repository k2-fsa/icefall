#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

# We don't download the LM file since it is so large that it will
# cause OOM error for CI later.
mkdir -p download/lm
pushd download/lm
wget -q http://www.openslr.org/resources/11/librispeech-vocab.txt
wget -q http://www.openslr.org/resources/11/librispeech-lexicon.txt
wget -q http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
ls -lh
gunzip librispeech-lm-norm.txt.gz

ls -lh
popd

pushd download/
wget -q https://huggingface.co/csukuangfj/librispeech-for-ci/resolve/main/LibriSpeech.tar.bz2
tar xf LibriSpeech.tar.bz2
rm LibriSpeech.tar.bz2

cd LibriSpeech
ln -s train-clean-100 train-clean-360
ln -s train-other-500 train-other-500
popd

mkdir -p data/manifests

lhotse prepare librispeech -j 2 -p dev-clean -p dev-other -p test-clean -p test-other -p train-clean-100 download/LibriSpeech data/manifests
ls -lh data/manifests

./local/compute_fbank_librispeech.py --dataset "dev-clean dev-other test-clean test-other train-clean-100" --perturb-speed False
ls -lh data/fbank

./prepare.sh --stage 5 --stop-stage 6

./zipformer/train.py \
  --world-size 1 \
  --num-epochs 1 \
  --start-epoch 1 \
  --use-fp16 0 \
  --exp-dir zipformer/exp-small \
  --causal 0 \
  --num-encoder-layers 1,1,1,1,1,1 \
  --feedforward-dim 64,96,96,96,96,96 \
  --encoder-dim 32,64,64,64,64,64 \
  --encoder-unmasked-dim 32,32,32,32,32,32 \
  --base-lr 0.04 \
  --full-libri 0 \
  --enable-musan 0 \
  --max-duration 30 \
  --print-diagnostics 1
