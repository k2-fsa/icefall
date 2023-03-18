#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/aishell/ASR

repo_url=https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2-2022-03-01

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./transducer_stateless_modified-2/pretrained.py \
    --method greedy_search \
    --max-sym-per-frame $sym \
    --checkpoint $repo/exp/pretrained.pt \
    --lang-dir $repo/data/lang_char \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

for method in modified_beam_search beam_search; do
  log "$method"

  ./transducer_stateless_modified-2/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --lang-dir $repo/data/lang_char \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done
