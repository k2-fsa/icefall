#!/usr/bin/env bash

set -ex

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

  ls -lh

  log "test pretrained.pt"

  for w in 1.wav 2.wav 3.wav; do
    log "test $w"
    python3 zipformer/pretrained.py \
      --checkpoint $repo/exp/pretrained.pt \
      --label-dict $repo/data/class_labels_indices.csv \
      $repo/test_wavs/$w
  done

}

test_pretrained
