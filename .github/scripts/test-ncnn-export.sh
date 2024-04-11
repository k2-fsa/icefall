#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

pushd egs/librispeech/ASR

log  "Install ncnn and pnnx"

# We are using a modified ncnn here. Will try to merge it to the official repo
# of ncnn
git clone https://github.com/csukuangfj/ncnn
pushd ncnn
git submodule init
git submodule update python/pybind11
python3 setup.py bdist_wheel
ls -lh dist/
pip install dist/*.whl
cd tools/pnnx
mkdir build
cd build

echo "which python3"

which python3
#/opt/hostedtoolcache/Python/3.8.16/x64/bin/python3

cmake -D Python3_EXECUTABLE=$(which python3) ..
make -j4 pnnx

./src/pnnx || echo "pass"

popd

export PATH=$PWD/ncnn/tools/pnnx/build/src:$PATH

log "=========================================================================="
repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained-epoch-30-avg-10-averaged.pt"

cd exp
ln -s pretrained-epoch-30-avg-10-averaged.pt epoch-99.pt
popd

log "Export via torch.jit.trace()"

./conv_emformer_transducer_stateless2/export-for-ncnn.py \
  --exp-dir $repo/exp \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32

pnnx $repo/exp/encoder_jit_trace-pnnx.pt
pnnx $repo/exp/decoder_jit_trace-pnnx.pt
pnnx $repo/exp/joiner_jit_trace-pnnx.pt

python3 ./conv_emformer_transducer_stateless2/streaming-ncnn-decode.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained-iter-468000-avg-16.pt"

cd exp
ln -s pretrained-iter-468000-avg-16.pt epoch-99.pt
popd

log "Export via torch.jit.trace()"

./lstm_transducer_stateless2/export-for-ncnn.py \
  --exp-dir $repo/exp \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0

pnnx $repo/exp/encoder_jit_trace-pnnx.pt
pnnx $repo/exp/decoder_jit_trace-pnnx.pt
pnnx $repo/exp/joiner_jit_trace-pnnx.pt

python3 ./lstm_transducer_stateless2/streaming-ncnn-decode.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
  $repo/test_wavs/1089-134686-0001.wav

python3 ./lstm_transducer_stateless2/ncnn-decode.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-99.pt
popd

./pruned_transducer_stateless7_streaming/export-for-ncnn.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --exp-dir $repo/exp \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  \
  --decode-chunk-len 32 \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,2048,2048,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512

pnnx $repo/exp/encoder_jit_trace-pnnx.pt
pnnx $repo/exp/decoder_jit_trace-pnnx.pt
pnnx $repo/exp/joiner_jit_trace-pnnx.pt

python3 ./pruned_transducer_stateless7_streaming/streaming-ncnn-decode.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
  $repo/test_wavs/1089-134686-0001.wav

rm -rf $repo
log "--------------------------------------------------------------------------"

log "=========================================================================="
repo_url=https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_char_bpe/L.pt"
git lfs pull --include "data/lang_char_bpe/L_disambig.pt"
git lfs pull --include "data/lang_char_bpe/Linv.pt"
git lfs pull --include "exp/pretrained.pt"

cd exp
ln -s pretrained.pt epoch-9999.pt
popd

./pruned_transducer_stateless7_streaming/export-for-ncnn-zh.py \
  --tokens $repo/data/lang_char_bpe/tokens.txt \
  --exp-dir $repo/exp \
  --use-averaged-model 0 \
  --epoch 9999 \
  --avg 1 \
  --decode-chunk-len 32 \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,1536,1536,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512

pnnx $repo/exp/encoder_jit_trace-pnnx.pt
pnnx $repo/exp/decoder_jit_trace-pnnx.pt
pnnx $repo/exp/joiner_jit_trace-pnnx.pt

python3 ./pruned_transducer_stateless7_streaming/streaming-ncnn-decode.py \
  --tokens $repo/data/lang_char_bpe/tokens.txt \
  --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
  --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
  --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
  --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
  --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
  --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
  $repo/test_wavs/0.wav

rm -rf $repo
log "--------------------------------------------------------------------------"
