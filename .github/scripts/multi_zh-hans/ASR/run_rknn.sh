#!/usr/bin/env bash

set -ex

python3 -m pip install kaldi-native-fbank soundfile librosa

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/multi_zh-hans/ASR



# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12-chinese
function export_2023_11_05() {
  d=exp
  mkdir $d
  pushd $d
  curl -SL -O https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main/data/lang_bpe_2000/tokens.txt
  curl -SL -O https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -o 0.wav https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main/test_wavs/DEV_T0000000000.wav
  curl -SL -o 1.wav https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main/test_wavs/DEV_T0000000001.wav
  curl -SL -o 2.wav https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/resolve/main/test_wavs/DEV_T0000000002.wav
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
    --chunk-size 32 \
    --left-context-frames 128 \
    --causal 1

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    dst=sherpa-onnx-$platform-streaming-zipformer-multi-zh-hans-2023-12-12
    mkdir -p $dst

    ./zipformer/export_rknn_transducer_streaming.py \
      --in-encoder $d/encoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1-chunk-32-left-128.onnx \
      --out-encoder $dst/encoder.rknn \
      --out-decoder $dst/decoder.rknn \
      --out-joiner $dst/joiner.rknn \
      --target-platform $platform

    cp $d/tokens.txt $dst
    mkdir $dst/test_wavs
    cp $d/*.wav $dst/test_wavs

    tar cjvf $dst.tar.bz2 $dst
    ls -lh $dst.tar.bz2
    mv $dst.tar.bz2 /icefall/
    ls -lh $dst/
    echo "---"

    rm -rf $dst
  done
}

export_2023_11_05
