#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/swbd/ASR

repo_url=https://huggingface.co/zrjin/icefall-asr-swbd-conformer-ctc-2023-8-26

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)


log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
ln -s epoch-98.pt epoch-99.pt
popd

ls -lh $repo/exp/*.pt

for method in ctc-decoding 1best; do
  log "$method"

  ./conformer_ctc/pretrained.py \
    --method $method \
    --checkpoint $repo/exp/epoch-99.pt \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    --words-file $repo/data/lang_bpe_500/words.txt \
    --HLG  $repo/data/lang_bpe_500/HLG.pt \
    --G $repo/data/lm/G_4_gram.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav
done
