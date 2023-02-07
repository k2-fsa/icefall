#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

log "=========================================================================="
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

log "Export via torch.jit.trace()"

./pruned_transducer_stateless7_streaming/jit_trace_export.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

log "Test exporting to ONNX format"

./pruned_transducer_stateless7_streaming/export-onnx.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

ls -lh $repo/exp

log "Run onnx_check.py"

./pruned_transducer_stateless7_streaming/onnx_check.py \
  --jit-encoder-filename $repo/exp/encoder_jit_trace.pt \
  --jit-decoder-filename $repo/exp/decoder_jit_trace.pt \
  --jit-joiner-filename $repo/exp/joiner_jit_trace.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx

log "Run onnx_pretrained.py"

./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"
