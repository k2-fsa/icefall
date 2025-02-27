#!/usr/bin/env bash

set -ex

python3 -m pip install kaldi-native-fbank soundfile librosa

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR


# https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed
# sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
function export_bilingual_zh_en() {
  d=exp_zh_en

  mkdir $d
  pushd $d

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/data/lang_char_bpe/tokens.txt

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/2.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/3.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-chinese-english-mixed/resolve/main/test_wavs/4.wav
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
    --decode-chunk-len 64 \
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

  ls -lh $d/

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
    $d/1.wav

  mkdir -p /icefall/rknn-models

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    mkdir -p $platform

    ./pruned_transducer_stateless7_streaming/export_rknn.py \
      --in-encoder $d/encoder-epoch-99-avg-1.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1.onnx \
      --out-encoder $platform/encoder.rknn \
      --out-decoder $platform/decoder.rknn \
      --out-joiner $platform/joiner.rknn \
      --target-platform $platform  2>/dev/null

    ls -lh $platform/

    ./pruned_transducer_stateless7_streaming/test_rknn_on_cpu_simulator.py \
      --encoder $d/encoder-epoch-99-avg-1.onnx \
      --decoder $d/decoder-epoch-99-avg-1.onnx \
      --joiner $d/joiner-epoch-99-avg-1.onnx \
      --tokens $d/tokens.txt \
      --wav $d/0.wav

    cp $d/tokens.txt $platform
    cp $d/*.wav $platform

    cp -av $platform /icefall/rknn-models
  done

  ls -lh /icefall/rknn-models
}

# https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t
# sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16
function export_bilingual_zh_en_small() {
  d=exp_zh_en_small

  mkdir $d
  pushd $d

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/data/lang_char_bpe/tokens.txt
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/test_wavs/2.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/test_wavs/3.wav
  curl -SL -O https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t/resolve/main/test_wavs/4.wav

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
    --decode-chunk-len 64 \
    \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dims 768,768,768,768,768 \
    --nhead 4,4,4,4,4 \
    --encoder-dims 256,256,256,256,256 \
    --attention-dims 192,192,192,192,192 \
    --encoder-unmasked-dims 192,192,192,192,192 \
    \
    --zipformer-downsampling-factors "1,2,4,8,2" \
    --cnn-module-kernels "31,31,31,31,31" \
    --decoder-dim 512 \
    --joiner-dim 512

  ls -lh $d/

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
    $d/1.wav

  mkdir -p /icefall/rknn-models-small

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    mkdir -p $platform

    ./pruned_transducer_stateless7_streaming/export_rknn.py \
      --in-encoder $d/encoder-epoch-99-avg-1.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1.onnx \
      --out-encoder $platform/encoder.rknn \
      --out-decoder $platform/decoder.rknn \
      --out-joiner $platform/joiner.rknn \
      --target-platform $platform  2>/dev/null

    ls -lh $platform/

    ./pruned_transducer_stateless7_streaming/test_rknn_on_cpu_simulator.py \
      --encoder $d/encoder-epoch-99-avg-1.onnx \
      --decoder $d/decoder-epoch-99-avg-1.onnx \
      --joiner $d/joiner-epoch-99-avg-1.onnx \
      --tokens $d/tokens.txt \
      --wav $d/0.wav

    cp $d/tokens.txt $platform
    cp $d/*.wav $platform

    cp -av $platform /icefall/rknn-models-small
  done

  ls -lh /icefall/rknn-models-small
}

export_bilingual_zh_en_small

export_bilingual_zh_en
