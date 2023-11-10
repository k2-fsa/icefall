#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://github.com/csukuangfj/icefall-asr-conformer-ctc-bpe-500
git lfs install

log "Downloading pre-trained model from $repo_url"
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.flac

log "CTC decoding"

./conformer_ctc/pretrained.py \
  --method ctc-decoding \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac

log "HLG decoding"

./conformer_ctc/pretrained.py \
  --method 1best \
  --num-classes 500 \
  --checkpoint $repo/exp/pretrained.pt \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --words-file $repo/data/lang_bpe_500/words.txt \
  --HLG $repo/data/lang_bpe_500/HLG.pt \
  $repo/test_wavs/1089-134686-0001.flac \
  $repo/test_wavs/1221-135766-0001.flac \
  $repo/test_wavs/1221-135766-0002.flac
