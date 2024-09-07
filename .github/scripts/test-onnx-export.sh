#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

log "=========================================================================="
repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
popd

log "Export via torch.jit.script()"
./zipformer/export.py \
  --use-averaged-model 0 \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --jit 1

log "Test export to ONNX format"
./zipformer/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal False \
  --chunk-size "16,32,64,-1" \
  --left-context-frames "64,128,256,-1"

ls -lh $repo/exp

log "Run onnx_check.py"

./zipformer/onnx_check.py \
  --jit-filename $repo/exp/jit_script.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx

log "Run onnx_pretrained.py"

./zipformer/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17
log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

log "Test export streaming model to ONNX format"
./zipformer/export-onnx-streaming.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 64

ls -lh $repo/exp

log "Run onnx_pretrained-streaming.py"

./zipformer/onnx_pretrained-streaming.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1-chunk-16-left-64.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1-chunk-16-left-64.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1-chunk-16-left-64.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo

log "--------------------------------------------------------------------------"

log "=========================================================================="
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
  --tokens $repo/data/lang_bpe_500/tokens.txt \
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

log "=========================================================================="
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-iter-1224000-avg-14.pt"

cd exp
ln -s pretrained-iter-1224000-avg-14.pt epoch-9999.pt
popd

log "Export via torch.jit.script()"

./pruned_transducer_stateless3/export.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 9999 \
  --avg 1 \
  --exp-dir $repo/exp/ \
  --jit 1

log "Test exporting to ONNX format"

./pruned_transducer_stateless3/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 9999 \
  --avg 1 \
  --exp-dir $repo/exp/

ls -lh $repo/exp

log "Run onnx_check.py"

./pruned_transducer_stateless3/onnx_check.py \
  --jit-filename $repo/exp/cpu_jit.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-9999-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-9999-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-9999-avg-1.onnx

log "Run onnx_pretrained.py"

./pruned_transducer_stateless3/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-9999-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-9999-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-9999-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless5-2022-05-13
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-epoch-39-avg-7.pt"

cd exp
ln -s pretrained-epoch-39-avg-7.pt epoch-99.pt
popd

log "Export via torch.jit.script()"

./pruned_transducer_stateless5/export.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir $repo/exp \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --jit 1

log "Test exporting to ONNX format"

./pruned_transducer_stateless5/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir $repo/exp \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512

ls -lh $repo/exp

log "Run onnx_check.py"

./pruned_transducer_stateless5/onnx_check.py \
  --jit-filename $repo/exp/cpu_jit.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx

log "Run onnx_pretrained.py"

./pruned_transducer_stateless5/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=

rm -rf $repo
log "--------------------------------------------------------------------------"
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

log "Export via torch.jit.script()"

./pruned_transducer_stateless7/export.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --feedforward-dims "1024,1024,2048,2048,1024" \
  --jit 1

log "Test exporting to ONNX format"

./pruned_transducer_stateless7/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --feedforward-dims "1024,1024,2048,2048,1024"

ls -lh $repo/exp

log "Run onnx_check.py"

./pruned_transducer_stateless7/onnx_check.py \
  --jit-filename $repo/exp/cpu_jit.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx

log "Run onnx_pretrained.py"

./pruned_transducer_stateless7/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

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

log "Test exporting to ONNX format"

./conv_emformer_transducer_stateless2/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32

log "Run onnx_pretrained.py"

./conv_emformer_transducer_stateless2/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1221-135766-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained-iter-468000-avg-16.pt"

cd exp
ln -s pretrained-iter-468000-avg-16.pt epoch-99.pt
popd

log "Export via torch.jit.trace()"

./lstm_transducer_stateless2/export.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp/ \
  --jit-trace 1

log "Test exporting to ONNX format"

./lstm_transducer_stateless2/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp

ls -lh $repo/exp

log "Run onnx_check.py"

./lstm_transducer_stateless2/onnx_check.py \
  --jit-encoder-filename $repo/exp/encoder_jit_trace.pt \
  --jit-decoder-filename $repo/exp/decoder_jit_trace.pt \
  --jit-joiner-filename $repo/exp/joiner_jit_trace.pt \
  --onnx-encoder-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --onnx-decoder-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --onnx-joiner-filename $repo/exp/joiner-epoch-99-avg-1.onnx

log "Run onnx_pretrained.py"

./lstm_transducer_stateless2/onnx_pretrained.py \
  --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
  --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
  --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  $repo/test_wavs/1221-135766-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"
