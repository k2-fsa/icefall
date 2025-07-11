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
function export_2023_02_20() {
  d=exp_2023_02_20

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

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    dst=sherpa-onnx-$platform-streaming-zipformer-bilingual-zh-en-2023-02-20
    mkdir -p $dst

    ./pruned_transducer_stateless7_streaming/export_rknn.py \
      --in-encoder $d/encoder-epoch-99-avg-1.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1.onnx \
      --out-encoder $dst/encoder.rknn \
      --out-decoder $dst/decoder.rknn \
      --out-joiner $dst/joiner.rknn \
      --target-platform $platform  2>/dev/null

    ls -lh $dst/

    ./pruned_transducer_stateless7_streaming/test_rknn_on_cpu_simulator.py \
      --encoder $d/encoder-epoch-99-avg-1.onnx \
      --decoder $d/decoder-epoch-99-avg-1.onnx \
      --joiner $d/joiner-epoch-99-avg-1.onnx \
      --tokens $d/tokens.txt \
      --wav $d/0.wav

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

# https://huggingface.co/csukuangfj/k2fsa-zipformer-bilingual-zh-en-t
# sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16
function export_2023_02_16() {
  d=exp_2023_02_16

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

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    dst=sherpa-onnx-$platform-streaming-zipformer-small-bilingual-zh-en-2023-02-16
    mkdir -p $dst

    ./pruned_transducer_stateless7_streaming/export_rknn.py \
      --in-encoder $d/encoder-epoch-99-avg-1.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1.onnx \
      --out-encoder $dst/encoder.rknn \
      --out-decoder $dst/decoder.rknn \
      --out-joiner $dst/joiner.rknn \
      --target-platform $platform  2>/dev/null

    ls -lh $dst/

    ./pruned_transducer_stateless7_streaming/test_rknn_on_cpu_simulator.py \
      --encoder $d/encoder-epoch-99-avg-1.onnx \
      --decoder $d/decoder-epoch-99-avg-1.onnx \
      --joiner $d/joiner-epoch-99-avg-1.onnx \
      --tokens $d/tokens.txt \
      --wav $d/0.wav

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

# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english
function export_2023_06_26() {
  d=exp_2023_06_26

  mkdir $d
  pushd $d

  curl -SL -O https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/exp/pretrained.pt
  mv pretrained.pt epoch-99.pt

  curl -SL -O https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/data/lang_bpe_500/tokens.txt

  curl -SL -o 0.wav https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/data/lang_bpe_500/tokens.txt
  curl -SL -o 1.wav https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/test_wavs/1221-135766-0001.wav
  curl -SL -o 2.wav https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17/resolve/main/test_wavs/1221-135766-0002.wav

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
    --chunk-size 32 \
    --left-context-frames 128 \
    --causal 1

  ls -lh $d/

  for platform in rk3562 rk3566 rk3568 rk3576 rk3588; do
    dst=sherpa-onnx-$platform-streaming-zipformer-en-2023-06-26
    mkdir -p $dst

    ./zipformer/export_rknn_transducer_streaming.py \
      --in-encoder $d/encoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-decoder $d/decoder-epoch-99-avg-1-chunk-32-left-128.onnx \
      --in-joiner $d/joiner-epoch-99-avg-1-chunk-32-left-128.onnx \
      --out-encoder $dst/encoder.rknn \
      --out-decoder $dst/decoder.rknn \
      --out-joiner $dst/joiner.rknn \
      --target-platform $platform

    ls -lh $dst/

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

if [[ $rknn_toolkit2_version == "2.1.0" ]]; then
  export_2023_02_16
  export_2023_02_20
else
  export_2023_06_26
fi
