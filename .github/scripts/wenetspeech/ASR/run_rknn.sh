#!/usr/bin/env bash

set -ex

python3 -m pip install kaldi-native-fbank soundfile librosa

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/wenetspeech/ASR

#https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#k2-fsa-icefall-asr-zipformer-wenetspeech-streaming-small-chinese
function export_zh_small() {
  d=exp_zh_small
  mkdir $d
  pushd $d
  curl -SL -O https://huggingface.co/k2-fsa/icefall-asr-zipformer-wenetspeech-streaming-small/resolve/main/data/lang_char/tokens.txt
  curl -SL -O https://huggingface.co/k2-fsa/icefall-asr-zipformer-wenetspeech-streaming-small/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -o 0.wav https://huggingface.co/k2-fsa/icefall-asr-zipformer-wenetspeech-streaming-small/resolve/main/test_wavs/DEV_T0000000000.wav
  curl -SL -o 1.wav https://huggingface.co/k2-fsa/icefall-asr-zipformer-wenetspeech-streaming-small/resolve/main/test_wavs/DEV_T0000000001.wav
  curl -SL -o 2.wav https://huggingface.co/k2-fsa/icefall-asr-zipformer-wenetspeech-streaming-small/resolve/main/test_wavs/DEV_T0000000002.wav
  ls -lh
  popd

  ./zipformer/export-onnx-streaming.py \
    --dynamic-batch 0 \
    --enable-int8-quantization 0 \
    --tokens $d/tokens.txt \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --exp-dir $d \
    --use-ctc 0 \
    --use-transducer 1 \
    \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    \
    --chunk-size 32 \
    --left-context-frames 128 \
    --causal 1

  out=/icefall/rknn-models-small-wenetspeech
  mkdir -p $out

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    mkdir -p $out/$platform

    ./zipformer/export_rknn_transducer_streaming.py \
      --in-encoder $d/encoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1-chunk-32-left-128.onnx \
      --out-encoder $out/$platform/encoder.rknn \
      --out-decoder $out/$platform/decoder.rknn \
      --out-joiner $out/$platform/joiner.rknn \
      --target-platform $platform

    cp $d/tokens.txt $out/$platform
    cp $d/*.wav $out/$platform
    ls -lh $out/$platform/
  done
  ls -h $out
  echo "---"
  ls -h $out/*
}

export_zh_small
