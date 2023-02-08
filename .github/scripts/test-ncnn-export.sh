#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

log "=========================================================================="
repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-epoch-30-avg-10-averaged.pt"

cd exp
ln -s pretrained-epoch-30-avg-10-averaged.pt epoch-99.pt
popd

log "Export via torch.jit.trace()"

./conv_emformer_transducer_stateless2/export-for-ncnn.py \
  --exp-dir $repo/exp \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32

cd $repo/exp

./ncnn/tools/pnnx/build/src/pnnx $repo/exp/encoder_jit_trace-pnnx.pt
./ncnn/tools/pnnx/build/src/pnnx $repo/exp/decoder_jit_trace-pnnx.pt
./ncnn/tools/pnnx/build/src/pnnx $repo/exp/joiner_jit_trace-pnnx.pt

rm -rf $repo
log "--------------------------------------------------------------------------"
