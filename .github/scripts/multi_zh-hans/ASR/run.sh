#!/usr/bin/env bash

set -ex

git config --global user.name "k2-fsa"
git config --global user.email "csukuangfj@gmail.com"
git config --global lfs.allowincompletepush true

python3 -m pip install onnxmltools==1.13.0 onnx==1.17.0 onnxruntime==1.17.1 sherpa-onnx

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/multi_zh-hans/ASR

log "pwd: $PWD"

function run_2023_9_2() {
  repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-2023-9-2
  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)
  pushd $repo
  cd exp
  git lfs pull --include pretrained.pt
  ln -s pretrained.pt epoch-99.pt
  cd ../data/lang_bpe_2000
  ls -lh
  git lfs pull --include L.pt L_disambig.pt Linv.pt bpe.model
  git lfs pull --include "*.model"
  ls -lh
  popd

  log "--------------------------------------------"
  log "Export non-streaming ONNX transducer models "
  log "--------------------------------------------"
  ./zipformer/export-onnx.py \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    --use-averaged-model 0 \
    --epoch 99 \
    --avg 1 \
    --exp-dir $repo/exp \
    --causal False \
    --fp16 1

  ls -lh $repo/exp

  ./zipformer/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav \
    $repo/test_wavs/DEV_T0000000001.wav \
    $repo/test_wavs/DEV_T0000000002.wav \
    $repo/test_wavs/TEST_MEETING_T0000000113.wav \
    $repo/test_wavs/TEST_MEETING_T0000000219.wav \
    $repo/test_wavs/TEST_MEETING_T0000000351.wav

  ./zipformer/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.int8.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.int8.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav \
    $repo/test_wavs/DEV_T0000000001.wav \
    $repo/test_wavs/DEV_T0000000002.wav \
    $repo/test_wavs/TEST_MEETING_T0000000113.wav \
    $repo/test_wavs/TEST_MEETING_T0000000219.wav \
    $repo/test_wavs/TEST_MEETING_T0000000351.wav

  ./zipformer/onnx_pretrained.py \
    --encoder-model-filename $repo/exp/encoder-epoch-99-avg-1.fp16.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-99-avg-1.fp16.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-99-avg-1.fp16.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav \
    $repo/test_wavs/DEV_T0000000001.wav \
    $repo/test_wavs/DEV_T0000000002.wav \
    $repo/test_wavs/TEST_MEETING_T0000000113.wav \
    $repo/test_wavs/TEST_MEETING_T0000000219.wav \
    $repo/test_wavs/TEST_MEETING_T0000000351.wav

  rm -rf $repo
}

function run_2023_11_05_streaming() {
  repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05
  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  pushd $repo
  cd exp/
  git lfs pull --include pretrained.pt
  rm -fv epoch-20.pt
  rm -fv *.onnx
  ln -s pretrained.pt epoch-20.pt
  cd ../data/lang_bpe_2000
  ls -lh
  git lfs pull --include L.pt L_disambig.pt Linv.pt bpe.model
  git lfs pull --include "*.model"
  ls -lh
  popd

  log "----------------------------------------"
  log "Export streaming ONNX CTC models "
  log "----------------------------------------"
  ./zipformer/export-onnx-streaming-ctc.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 20 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 1 \
    --fp16 1

  ls -lh $repo/exp/

  log "------------------------------------------------------------"
  log "Test exported streaming ONNX CTC models (greedy search)     "
  log "------------------------------------------------------------"

  test_wavs=(
    DEV_T0000000000.wav
    DEV_T0000000001.wav
    DEV_T0000000002.wav
    TEST_MEETING_T0000000113.wav
    TEST_MEETING_T0000000219.wav
    TEST_MEETING_T0000000351.wav
  )

  for w in ${test_wavs[@]}; do
    log "----fp32----"
    ./zipformer/onnx_pretrained-streaming-ctc.py \
      --model-filename $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
      --tokens $repo/data/lang_bpe_2000/tokens.txt \
      $repo/test_wavs/$w

    log "----int8----"

    ./zipformer/onnx_pretrained-streaming-ctc.py \
      --model-filename $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
      --tokens $repo/data/lang_bpe_2000/tokens.txt \
      $repo/test_wavs/$w

    log "----fp16----"

    ./zipformer/onnx_pretrained-streaming-ctc.py \
      --model-filename $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.fp16.onnx \
      --tokens $repo/data/lang_bpe_2000/tokens.txt \
      $repo/test_wavs/$w
  done

  log "Upload onnx CTC models to huggingface"
  name=(
    sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13
    sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-int8-2023-12-13
    sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-fp16-2023-12-13
    )
  for n in ${name[@]}; do
      url=https://huggingface.co/k2-fsa/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13 ]]; then
        cp -v $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-int8-2023-12-13 ]]; then
        cp -v $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.int8.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-fp16-2023-12-13 ]]; then
        cp -v $repo/exp/ctc-epoch-20-avg-1-chunk-16-left-128.fp16.onnx $dst
      fi

      cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
      cp -v $repo/data/lang_bpe_2000/bpe.model $dst
      mkdir -p $dst/test_wavs
      cp -v $repo/test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "bpe.model" "*.wav"
      ls -lh
      file bpe.model
      git status
      git add .
      git commit -m "upload model" && git push https://k2-fsa:${HF_TOKEN}@huggingface.co/k2-fsa/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done

  log "----------------------------------------"
  log "Export streaming ONNX transducer models "
  log "----------------------------------------"

  ./zipformer/export-onnx-streaming.py \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 20 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 0 \
    --fp16 1

  ls -lh $repo/exp

  log "------------------------------------------------------------"
  log "Test exported streaming ONNX transducer models (Python code)"
  log "------------------------------------------------------------"

  log "test fp32"
  ./zipformer/onnx_pretrained-streaming.py \
    --encoder-model-filename $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav

  log "test int8"
  ./zipformer/onnx_pretrained-streaming.py \
    --encoder-model-filename $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav

  log "test fp16"
  ./zipformer/onnx_pretrained-streaming.py \
    --encoder-model-filename $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.fp16.onnx \
    --decoder-model-filename $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.fp16.onnx \
    --joiner-model-filename $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.fp16.onnx \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    $repo/test_wavs/DEV_T0000000000.wav

  name=(
    sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-13
    sherpa-onnx-streaming-zipformer-multi-zh-hans-int8-2023-12-13
    sherpa-onnx-streaming-zipformer-multi-zh-hans-fp16-2023-12-13
  )

  for n in ${name[@]}; do
      url=https://huggingface.co/csukuangfj/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-13 ]]; then
        cp -v $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.onnx $dst
        cp -v $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx $dst
        cp -v $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-multi-zh-hans-int8-2023-12-13 ]]; then
        cp -v $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx $dst
        cp -v $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.onnx $dst
        cp -v $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-multi-zh-hans-fp16-2023-12-13 ]]; then
        cp -v $repo/exp/encoder-epoch-20-avg-1-chunk-16-left-128.fp16.onnx $dst
        cp -v $repo/exp/decoder-epoch-20-avg-1-chunk-16-left-128.fp16.onnx $dst
        cp -v $repo/exp/joiner-epoch-20-avg-1-chunk-16-left-128.fp16.onnx $dst
      fi

      cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
      cp -v $repo/data/lang_bpe_2000/bpe.model $dst
      mkdir -p $dst/test_wavs
      cp -v $repo/test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "bpe.model" "*.wav"
      ls -lh
      file bpe.model
      git status
      git add .
      git commit -m "upload model" && git push https://csukuangfj:${HF_TOKEN}@huggingface.co/csukuangfj/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done
}

function run_2023_12_12_streaming() {
  log "Upload onnx transducer models to huggingface"

  url=https://huggingface.co/k2-fsa/sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12
  GIT_LFS_SKIP_SMUDGE=1 git clone $url
  dst=$(basename $url)
  cp -v $repo/exp/encoder*.onnx $dst
  cp -v $repo/exp/decoder*.onnx $dst
  cp -v $repo/exp/joiner*.onnx $dst
  cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
  cp -v $repo/data/lang_bpe_2000/bpe.model $dst
  mkdir -p $dst/test_wavs
  cp -v $repo/test_wavs/*.wav $dst/test_wavs
  cd $dst
  git lfs track "*.onnx" bpe.model "*.wav"
  git add .
  git commit -m "upload model" && git push https://k2-fsa:${HF_TOKEN}@huggingface.co/k2-fsa/$dst main || true

  log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
  rm -rf .git
  rm -fv .gitattributes
  cd ..
  tar cjfv $dst.tar.bz2 $dst
  ls -lh *.tar.bz2
  mv -v $dst.tar.bz2 ../../../
}

function run_yuekai_large() {
  repo_url=https://csukuangfj:${HF_TOKEN}@huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-large
  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)
  pushd $repo
  git lfs pull --include pretrained.pt
  mv pretrained.pt epoch-99.pt
  curl -SL -O https://huggingface.co/pingzxy/icefall-asr-multi-zh-hans-zipformer-large-onnx/resolve/main/tokens.txt
  popd

  log "----------------------------------------"
  log "Export streaming ONNX CTC models "
  log "----------------------------------------"
  ./zipformer/export-onnx-streaming-ctc.py \
    --exp-dir $repo/ \
    --tokens $repo/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 99 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 1 \
    \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 768,1024,1536,2048,1536,768 \
    --encoder-dim 256,384,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    \
    --fp16 1 \
    --use-whisper-features 1


  ls -lh $repo/
  pushd $repo

cat >README.md <<EOF
# Introduction

This model is converted
from
https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-large

The training code can be found at
https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#multi-chinese-datasets-char-based-training-results-streaming-on-zipformer-large-model
EOF

  mv -v ctc-epoch-99-avg-1-chunk-16-left-128.fp16.onnx model.fp16.onnx
  mv -v ctc-epoch-99-avg-1-chunk-16-left-128.int8.onnx model.int8.onnx
  mv -v ctc-epoch-99-avg-1-chunk-16-left-128.onnx model.onnx

  ls -lh *.onnx

  mkdir test_wavs
  cd test_wavs
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/8k.wav
  popd

  for w in 0.wav 1.wav 8k.wav; do
    log "---fp32---"
    sherpa-onnx \
      --zipformer2-ctc-model=$repo/model.onnx \
      --tokens=$repo/tokens.txt \
      $repo/test_wavs/$w

    log "---int8---"

    sherpa-onnx \
      --zipformer2-ctc-model=$repo/model.int8.onnx \
      --tokens=$repo/tokens.txt \
      $repo/test_wavs/$w

    log "---fp16---"

    sherpa-onnx \
      --zipformer2-ctc-model=$repo/model.fp16.onnx \
      --tokens=$repo/tokens.txt \
      $repo/test_wavs/$w
  done

  name=(
    sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30
    sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30
    sherpa-onnx-streaming-zipformer-ctc-zh-fp16-2025-06-30
  )
  for n in ${name[@]}; do
      url=https://huggingface.co/csukuangfj/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-ctc-zh-2025-06-30 ]]; then
        cp -v $repo/model.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-ctc-zh-int8-2025-06-30 ]]; then
        cp -v $repo/model.int8.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-ctc-zh-fp16-2025-06-30 ]]; then
        cp -v $repo/model.fp16.onnx $dst
      fi

      cp -v $repo/tokens.txt $dst
      cp -v $repo/README.md $dst
      mkdir -p $dst/test_wavs
      cp -v $repo/test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "*.wav"
      ls -lh
      git status
      git add .
      git commit -m "upload model" && git push https://csukuangfj:${HF_TOKEN}@huggingface.co/csukuangfj/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done

  rm $repo/*.onnx

  log "----------------------------------------"
  log "Export streaming ONNX transducer models "
  log "----------------------------------------"

  ./zipformer/export-onnx-streaming.py \
    --exp-dir $repo \
    --tokens $repo/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 99 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 0 \
    \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 768,1024,1536,2048,1536,768 \
    --encoder-dim 256,384,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    \
    --fp16 1 \
    --use-whisper-features 1

  ls -lh $repo
  pushd $repo
  for m in encoder decoder joiner; do
    mv -v $m-epoch-99-avg-1-chunk-16-left-128.onnx $m.onnx
    mv -v $m-epoch-99-avg-1-chunk-16-left-128.fp16.onnx $m.fp16.onnx
    mv -v $m-epoch-99-avg-1-chunk-16-left-128.int8.onnx $m.int8.onnx
  done
  ls -lh *.onnx
  popd

  for w in 0.wav 1.wav 8k.wav; do
    log "---fp32---"
      sherpa-onnx \
        --encoder=$repo/encoder.onnx \
        --decoder=$repo/decoder.onnx \
        --joiner=$repo/joiner.onnx \
        --tokens=$repo/tokens.txt \
        $repo/test_wavs/$w

    log "---int8---"

      sherpa-onnx \
        --encoder=$repo/encoder.int8.onnx \
        --decoder=$repo/decoder.onnx \
        --joiner=$repo/joiner.int8.onnx \
        --tokens=$repo/tokens.txt \
        $repo/test_wavs/$w

    log "---fp16---"

      sherpa-onnx \
        --encoder=$repo/encoder.fp16.onnx \
        --decoder=$repo/decoder.fp16.onnx \
        --joiner=$repo/joiner.fp16.onnx \
        --tokens=$repo/tokens.txt \
        $repo/test_wavs/$w
  done

  name=(
    sherpa-onnx-streaming-zipformer-zh-2025-06-30
    sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30
    sherpa-onnx-streaming-zipformer-zh-fp16-2025-06-30
  )
  for n in ${name[@]}; do
      url=https://huggingface.co/csukuangfj/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-zh-2025-06-30 ]]; then
        cp -v $repo/encoder.onnx $dst
        cp -v $repo/decoder.onnx $dst
        cp -v $repo/joiner.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30 ]]; then
        cp -v $repo/encoder.int8.onnx $dst
        cp -v $repo/decoder.onnx $dst
        cp -v $repo/joiner.int8.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-zh-fp16-2025-06-30 ]]; then
        cp -v $repo/encoder.fp16.onnx $dst
        cp -v $repo/decoder.fp16.onnx $dst
        cp -v $repo/joiner.fp16.onnx $dst
      fi

      cp -v $repo/tokens.txt $dst
      cp -v $repo/README.md $dst
      mkdir -p $dst/test_wavs
      cp -v $repo/test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "*.wav"
      ls -lh
      git status
      git add .
      git commit -m "upload model" && git push https://csukuangfj:${HF_TOKEN}@huggingface.co/csukuangfj/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done
}

function run_yuekai_xl() {
  repo_url=https://csukuangfj:${HF_TOKEN}@huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-xl
  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)

  pushd $repo
  git lfs pull --include pretrained.pt
  git lfs pull --include data/lang_bpe_2000/bpe.model
  mv pretrained.pt epoch-99.pt
  ls -lh *.pt
  popd

  log "----------------------------------------"
  log "Export streaming ONNX CTC models "
  log "----------------------------------------"
  ./zipformer/export-onnx-streaming-ctc.py \
    --exp-dir $repo/ \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 99 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 1 \
    \
    --num-encoder-layers 2,3,5,6,5,3 \
    --feedforward-dim 1536,2048,3072,4096,3072,1536 \
    --encoder-dim 512,768,1024,1536,1024,512 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --decoder-dim 768 --joiner-dim 768 \
    --value-head-dim 18 \
    --query-head-dim 48 \
    --num-heads 4,4,4,8,4,4 \
    \
    --fp16 1 \
    --use-whisper-features 1 \
    --use-external-data 1

  mv -v ctc-epoch-99-avg-1-chunk-16-left-128.int8.onnx model.int8.onnx
  mv -v ctc-epoch-99-avg-1-chunk-16-left-128.fp16.onnx model.fp16.onnx

  ls -lh *.onnx

  mkdir test_wavs
  pushd test_wavs
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/0.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/1.wav
  curl -SL -O https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/resolve/main/test_wavs/8k.wav
  popd

  for w in 0.wav 1.wav 8k.wav; do
    log "---int8---"

    sherpa-onnx \
      --zipformer2-ctc-model=./model.int8.onnx \
      --tokens=$repo/data/lang_bpe_2000/tokens.txt \
      test_wavs/$w

    log "---fp16---"

    sherpa-onnx \
      --zipformer2-ctc-model=./model.fp16.onnx \
      --tokens=$repo/data/lang_bpe_2000/tokens.txt \
      test_wavs/$w
  done

  pushd $repo
cat >README.md <<EOF
# Introduction

This model is converted
from
https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-xl

The training code can be found at
https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#multi-chinese-datasets-char-based-training-results-streaming-on-zipformer-xl-model
EOF
  popd

  name=(
    sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30
    sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-fp16-2025-06-30
  )

  for n in ${name[@]}; do
      url=https://huggingface.co/csukuangfj/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-fp16-2025-06-30 ]]; then
        cp -v model.fp16.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-ctc-zh-xlarge-int8-2025-06-30 ]]; then
        cp -v model.int8.onnx $dst
      fi

      cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
      cp -v $repo/data/lang_bpe_2000/bpe.model $dst
      cp -v $repo/README.md $dst
      mkdir -p $dst/test_wavs
      cp -v ./test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "*.wav" "bpe.model"
      ls -lh
      git status
      git add .
      git commit -m "upload model" && git push https://csukuangfj:${HF_TOKEN}@huggingface.co/csukuangfj/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..

      ls -lh $dst
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done

  rm -fv *.onnx *.weights

  log "----------------------------------------"
  log "Export streaming ONNX transducer models "
  log "----------------------------------------"

  ./zipformer/export-onnx-streaming.py \
    --exp-dir $repo/ \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
    --causal 1 \
    --avg 1 \
    --epoch 99 \
    --use-averaged-model 0 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --use-ctc 0 \
    \
    --num-encoder-layers 2,3,5,6,5,3 \
    --feedforward-dim 1536,2048,3072,4096,3072,1536 \
    --encoder-dim 512,768,1024,1536,1024,512 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --decoder-dim 768 --joiner-dim 768 \
    --value-head-dim 18 \
    --query-head-dim 48 \
    --num-heads 4,4,4,8,4,4 \
    \
    --fp16 1 \
    --use-whisper-features 1 \
    --use-external-data 1

    ls -lh *.onnx
    ls -lh *.weights

    mv encoder-epoch-99-avg-1-chunk-16-left-128.fp16.onnx encoder.fp16.onnx
    mv encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx encoder.int8.onnx

    mv $repo/decoder-epoch-99-avg-1-chunk-16-left-128.onnx decoder.onnx
    mv $repo/decoder-epoch-99-avg-1-chunk-16-left-128.fp16.onnx decoder.fp16.onnx

    mv $repo/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx joiner.int8.onnx
    mv $repo/joiner-epoch-99-avg-1-chunk-16-left-128.fp16.onnx joiner.fp16.onnx

  name=(
    sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30
    sherpa-onnx-streaming-zipformer-zh-xlarge-fp16-2025-06-30
  )

  for n in ${name[@]}; do
      url=https://huggingface.co/csukuangfj/$n
      GIT_LFS_SKIP_SMUDGE=1 git clone $url
      dst=$(basename $url)
      if [[ $n == sherpa-onnx-streaming-zipformer-zh-xlarge-fp16-2025-06-30 ]]; then
        cp -v encoder.fp16.onnx $dst
        cp -v decoder.fp16.onnx $dst
        cp -v joiner.fp16.onnx $dst
      elif [[ $n == sherpa-onnx-streaming-zipformer-zh-xlarge-int8-2025-06-30 ]]; then
        cp -v encoder.int8.onnx $dst
        cp -v decoder.onnx $dst
        cp -v joiner.int8.onnx $dst
      fi

      cp -v $repo/data/lang_bpe_2000/tokens.txt $dst
      cp -v $repo/data/lang_bpe_2000/bpe.model $dst
      cp -v $repo/README.md $dst
      mkdir -p $dst/test_wavs
      cp -v ./test_wavs/*.wav $dst/test_wavs
      cd $dst
      git lfs track "*.onnx" "*.wav" "bpe.model"
      ls -lh
      git status
      git add .
      git commit -m "upload model" && git push https://csukuangfj:${HF_TOKEN}@huggingface.co/csukuangfj/$dst main || true

      log "Upload models to https://github.com/k2-fsa/sherpa-onnx"
      rm -rf .git
      rm -fv .gitattributes
      cd ..

      ls -lh $dst
      tar cjfv $dst.tar.bz2 $dst
      ls -lh *.tar.bz2
      mv -v $dst.tar.bz2 ../../../
  done

  rm -fv *.onnx *.weights
}

# run_yuekai_large
# run_yuekai_xl
# run_2023_9_2
run_2023_11_05_streaming
# run_2023_12_12_streaming
