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

for method in modified_beam_search greedy_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --context-size 1 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_char/tokens.txt \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
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


for method in modified_beam_search greedy_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --context-size 1 \
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


for method in modified_beam_search greedy_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --context-size 1 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_char/tokens.txt \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

