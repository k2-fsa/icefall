#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "data/lang_bpe_500/tokens.txt"
git lfs pull --include "exp/jit_script.pt"
git lfs pull --include "exp/pretrained.pt"
ln -s pretrained.pt epoch-99.pt
ls -lh *.pt
popd

log "Export to torchscript model"
./zipformer/export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --jit 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.script()"

./zipformer/jit_pretrained.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --nn-model-filename $repo/exp/jit_script.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

for method in greedy_search modified_beam_search fast_beam_search; do
  log "$method"

  ./zipformer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --tokens $repo/data/lang_bpe_500/tokens.txt \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p zipformer/exp
  ln -s $PWD/$repo/exp/pretrained.pt zipformer/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh zipformer/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "Decoding with $method"

    ./zipformer/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --exp-dir zipformer/exp
  done

  rm zipformer/exp/*.pt
fi
