#!/usr/bin/env bash

set -ex

python3 -m pip install onnxoptimizer onnxsim

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/audioset/AT

function test_pretrained() {
  repo_url=https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12
  repo=$(basename $repo_url)
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  pushd $repo/exp
  git lfs pull --include pretrained.pt
  ln -s pretrained.pt epoch-99.pt
  ls -lh
  popd

  log "test pretrained.pt"

  python3 zipformer/pretrained.py \
    --checkpoint $repo/exp/pretrained.pt \
    --label-dict $repo/data/class_labels_indices.csv \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/4.wav

  log "test jit export"
  ls -lh $repo/exp/
  python3 zipformer/export.py \
      --exp-dir $repo/exp \
      --epoch 99 \
      --avg 1 \
      --use-averaged-model 0 \
      --jit 1
  ls -lh $repo/exp/

  log "test jit models"
  python3 zipformer/jit_pretrained.py \
      --nn-model-filename $repo/exp/jit_script.pt \
      --label-dict $repo/data/class_labels_indices.csv \
      $repo/test_wavs/1.wav \
      $repo/test_wavs/2.wav \
      $repo/test_wavs/3.wav \
      $repo/test_wavs/4.wav

  log "test onnx export"
  ls -lh $repo/exp/
  python3 zipformer/export-onnx.py \
      --exp-dir $repo/exp \
      --epoch 99 \
      --avg 1 \
      --use-averaged-model 0

  ls -lh $repo/exp/

  pushd $repo/exp/
  mv model-epoch-99-avg-1.onnx model.onnx
  mv model-epoch-99-avg-1.int8.onnx model.int8.onnx
  popd

  ls -lh $repo/exp/

  log "test onnx models"
  for m in model.onnx model.int8.onnx; do
    log "$m"
    python3 zipformer/onnx_pretrained.py \
        --model-filename $repo/exp/model.onnx \
        --label-dict $repo/data/class_labels_indices.csv \
        $repo/test_wavs/1.wav \
        $repo/test_wavs/2.wav \
        $repo/test_wavs/3.wav \
        $repo/test_wavs/4.wav
  done

  log "prepare data for uploading to huggingface"
  dst=/icefall/model-onnx
  mkdir -p $dst
  cp -v $repo/exp/*.onnx $dst/
  cp -v $repo/data/* $dst/
  cp -av $repo/test_wavs $dst

  ls -lh $dst
  ls -lh $dst/test_wavs
}

test_pretrained
