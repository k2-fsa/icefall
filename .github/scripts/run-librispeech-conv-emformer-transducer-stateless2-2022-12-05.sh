#!/usr/bin/env bash
#
set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

log "Downloading pre-trained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)
pushd $repo
git lfs pull --include "exp/pretrained-epoch-30-avg-10-averaged.pt"
git lfs pull --include "data/lang_bpe_500/bpe.model"
cd exp
ln -s pretrained-epoch-30-avg-10-averaged.pt epoch-99.pt
popd

log "Display test files"
tree $repo/
soxi $repo/test_wavs/*.wav
ls -lh $repo/test_wavs/*.wav

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
cmake -D Python3_EXECUTABLE=/opt/hostedtoolcache/Python/3.8.14/x64/bin/python3 ..
make -j4 pnnx

./src/pnnx || echo "pass"

popd

log "Test exporting to pnnx format"

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
  --memory-size 32 \

./ncnn/tools/pnnx/build/src/pnnx $repo/exp/encoder_jit_trace-pnnx.pt
./ncnn/tools/pnnx/build/src/pnnx $repo/exp/decoder_jit_trace-pnnx.pt
./ncnn/tools/pnnx/build/src/pnnx $repo/exp/joiner_jit_trace-pnnx.pt

./conv_emformer_transducer_stateless2/streaming-ncnn-decode.py \
 --bpe-model-filename $repo/data/lang_bpe_500/bpe.model \
 --encoder-param-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.param \
 --encoder-bin-filename $repo/exp/encoder_jit_trace-pnnx.ncnn.bin \
 --decoder-param-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.param \
 --decoder-bin-filename $repo/exp/decoder_jit_trace-pnnx.ncnn.bin \
 --joiner-param-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.param \
 --joiner-bin-filename $repo/exp/joiner_jit_trace-pnnx.ncnn.bin \
 $repo/test_wavs/1089-134686-0001.wav
