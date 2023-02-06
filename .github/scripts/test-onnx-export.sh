#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
popd

log "Test exporting to ONNX format"

./pruned_transducer_stateless7_streaming/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

ls -lh $repo/exp

./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \

rm -rfv $repo
