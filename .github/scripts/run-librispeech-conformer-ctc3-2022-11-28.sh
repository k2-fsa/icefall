#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-conformer-ctc3-2022-11-27

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
soxi $repo/test_wavs/*.wav
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
git lfs pull --include "data/*"
git lfs pull --include "exp/jit_trace.pt"
git lfs pull --include "exp/pretrained.pt"
ln -s pretrained.pt epoch-99.pt
ls -lh *.pt
popd

log "Decode with models exported by torch.jit.trace()"

for m in ctc-decoding 1best; do
  ./conformer_ctc3/jit_pretrained.py \
    --model-filename $repo/exp/jit_trace.pt \
    --words-file $repo/data/lang_bpe_500/words.txt  \
    --HLG $repo/data/lang_bpe_500/HLG.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --G $repo/data/lm/G_4_gram.pt \
    --method $m \
    --sample-rate 16000 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

log "Export to torchscript model"

./conformer_ctc3/export.py \
  --exp-dir $repo/exp \
  --lang-dir $repo/data/lang_bpe_500 \
  --jit-trace 1 \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.trace()"

for m in ctc-decoding 1best; do
  ./conformer_ctc3/jit_pretrained.py \
    --model-filename $repo/exp/jit_trace.pt \
    --words-file $repo/data/lang_bpe_500/words.txt  \
    --HLG $repo/data/lang_bpe_500/HLG.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --G $repo/data/lm/G_4_gram.pt \
    --method $m \
    --sample-rate 16000 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

for m in ctc-decoding 1best; do
  ./conformer_ctc3/pretrained.py \
    --checkpoint $repo/exp/pretrained.pt \
    --words-file $repo/data/lang_bpe_500/words.txt  \
    --HLG $repo/data/lang_bpe_500/HLG.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --G $repo/data/lm/G_4_gram.pt \
    --method $m \
    --sample-rate 16000 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p conformer_ctc3/exp
  ln -s $PWD/$repo/exp/pretrained.pt conformer_ctc3/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh conformer_ctc3/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in ctc-decoding 1best; do
    log "Decoding with $method"
    ./conformer_ctc3/decode.py \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --exp-dir conformer_ctc3/exp/ \
      --max-duration $max_duration \
      --decoding-method $method \
      --lm-dir data/lm
  done

  rm conformer_ctc3/exp/*.pt
fi
