#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/ksponspeech/ASR


function test_pretrained_non_streaming() {
  git lfs install
  git clone https://huggingface.co/johnBamma/icefall-asr-ksponspeech-zipformer-2024-06-24
  repo=icefall-asr-ksponspeech-zipformer-2024-06-24
  pushd $repo
  mkdir test_wavs
  cd test_wavs
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/2.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/3.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/trans.txt
  cd ../exp
  ln -s pretrained.pt epoch-99.pt
  ls -lh
  popd

  log 'test pretrained.py'
  ./zipformer/pretrained.py \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_5000/tokens.txt \
      --method greedy_search \
      $repo/test_wavs/0.wav \
      $repo/test_wavs/1.wav \
      $repo/test_wavs/2.wav \
      $repo/test_wavs/3.wav

  log 'test export-onnx.py'

  ./zipformer/export-onnx.py \
    --tokens $repo/data/lang_bpe_5000/tokens.txt \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --exp-dir $repo/exp/

  ls -lh $repo/exp

  ls -lh $repo/data/lang_bpe_5000/

  log 'test exported onnx models'
  ./zipformer/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
    --tokens $repo/data/lang_bpe_5000/tokens.txt \
    $repo/test_wavs/0.wav

  dst=/tmp/model-2024-06-24
  mkdir -p $dst

  cp -av $repo/test_wavs $dst
  cp -v $repo/exp/*.onnx $dst
  cp -v $repo/exp/*.onnx $dst
  cp -v $repo/data/lang_bpe_5000/tokens.txt $dst
  cp -v $repo/data/lang_bpe_5000/bpe.model $dst
  rm -rf $repo
}

function test_pretrained_streaming() {
  git lfs install
  git clone https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12
  repo=icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12
  pushd $repo
  mkdir test_wavs
  cd test_wavs
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/2.wav
  curl -SL -O https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-korean-2024-06-16/resolve/main/test_wavs/3.wav
  cd ../exp
  ln -s pretrained.pt epoch-99.pt
  ls -lh
  popd

  log 'test pretrained.py'
  ./pruned_transducer_stateless7_streaming/pretrained.py \
      --checkpoint $repo/exp/pretrained.pt \
      --tokens $repo/data/lang_bpe_5000/tokens.txt \
      --method greedy_search \
      $repo/test_wavs/0.wav \
      $repo/test_wavs/1.wav \
      $repo/test_wavs/2.wav \
      $repo/test_wavs/3.wav

  log 'test export-onnx.py'

  ./pruned_transducer_stateless7_streaming/export-onnx.py \
    --tokens $repo/data/lang_bpe_5000/tokens.txt \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --decode-chunk-len 32 \
    --exp-dir $repo/exp/

  ls -lh $repo/exp

  ls -lh $repo/data/lang_bpe_5000/

  log 'test exported onnx models'
  ./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
    --tokens $repo/data/lang_bpe_5000/tokens.txt \
    $repo/test_wavs/0.wav

  dst=/tmp/model-2024-06-16
  mkdir -p $dst

  cp -v $repo/exp/*.onnx $dst
  cp -v $repo/exp/*.onnx $dst
  cp -v $repo/data/lang_bpe_5000/tokens.txt $dst
  cp -v $repo/data/lang_bpe_5000/bpe.model $dst
  rm -rf $repo
}

test_pretrained_non_streaming
test_pretrained_streaming
