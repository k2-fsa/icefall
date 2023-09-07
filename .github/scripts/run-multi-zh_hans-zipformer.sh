#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/multi_zh-hans/ASR

repo_url=https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-2023-9-2/

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)


log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
ln -s epoch-20.pt epoch-99.pt
popd

ls -lh $repo/exp/*.pt


./zipformer/pretrained.py \
  --checkpoint $repo/exp/epoch-99.pt \
  --tokens $repo/data/lang_bpe_2000/tokens.txt \
  --method greedy_search \
$repo/test_wavs/DEV_T0000000000.wav \
$repo/test_wavs/DEV_T0000000001.wav \
$repo/test_wavs/DEV_T0000000002.wav

for method in modified_beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/epoch-99.pt \
    --tokens $repo/data/lang_bpe_2000/tokens.txt \
  $repo/test_wavs/DEV_T0000000000.wav \
  $repo/test_wavs/DEV_T0000000001.wav \
  $repo/test_wavs/DEV_T0000000002.wav
done
