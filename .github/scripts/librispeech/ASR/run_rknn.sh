#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR


# https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed
function export_bilingual_zh_en() {
  d=exp_zh_en

  mkdir $d
  pushd $d

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/data/lang_char_bpe/tokens.txt
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/data/lang_char_bpe/bpe.model

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/BAC009S0764W0164.wav
  ls -lh
  popd

  ./pruned_transducer_stateless7_streaming/export-onnx-zh.py \
    --dynamic-batch 0 \
    --enable-int8-quantization 0 \
    --tokens $d/tokens.txt \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --exp-dir $d/ \
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

  ./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
    --encoder-model-filename $d/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $d/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $d/joiner-epoch-99-avg-1.onnx \
    --tokens $d/tokens.txt \
    $d/0.wav

  ./pruned_transducer_stateless7_streaming/onnx_pretrained.py \
    --encoder-model-filename $d/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $d/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $d/joiner-epoch-99-avg-1.onnx \
    --tokens $d/tokens.txt \
    $d/BAC009S0764W0164.wav

  ls -lh $d/
}

export_bilingual_zh_en
