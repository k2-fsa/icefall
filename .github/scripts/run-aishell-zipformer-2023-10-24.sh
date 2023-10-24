#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/aishell/ASR

git lfs install

fbank_url=https://huggingface.co/csukuangfj/aishell-test-dev-manifests
log "Downloading pre-commputed fbank from $fbank_url"

git clone https://huggingface.co/csukuangfj/aishell-test-dev-manifests
ln -s $PWD/aishell-test-dev-manifests/data .

log "======================="
log "CI testing large model"
repo_url=https://huggingface.co/zrjin/icefall-asr-aishell-zipformer-large-2023-10-24/
log "Downloading pre-trained model from $repo_url"
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

./zipformer/pretrained.py \
  --method greedy_search \
  --max-sym-per-frame $sym \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_char/tokens.txt \
  $repo/test_wavs/BAC009S0764W0121.wav \
  $repo/test_wavs/BAC009S0764W0122.wav \
  $repo/test_wavs/BAC009S0764W0123.wav

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

log "======================="
log "CI testing medium model"
repo_url=https://huggingface.co/zrjin/icefall-asr-aishell-zipformer-2023-10-24/
log "Downloading pre-trained model from $repo_url"
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

./zipformer/pretrained.py \
  --method greedy_search \
  --max-sym-per-frame $sym \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_char/tokens.txt \
  $repo/test_wavs/BAC009S0764W0121.wav \
  $repo/test_wavs/BAC009S0764W0122.wav \
  $repo/test_wavs/BAC009S0764W0123.wav

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done


log "======================="
log "CI testing small model"
repo_url=https://huggingface.co/zrjin/icefall-asr-aishell-zipformer-small-2023-10-24/
log "Downloading pre-trained model from $repo_url"
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

./zipformer/pretrained.py \
  --method greedy_search \
  --max-sym-per-frame $sym \
  --checkpoint $repo/exp/pretrained.pt \
  --tokens $repo/data/lang_char/tokens.txt \
  $repo/test_wavs/BAC009S0764W0121.wav \
  $repo/test_wavs/BAC009S0764W0122.wav \
  $repo/test_wavs/BAC009S0764W0123.wav

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

