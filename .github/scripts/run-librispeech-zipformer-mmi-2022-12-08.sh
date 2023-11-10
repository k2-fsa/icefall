#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-mmi-2022-12-08

log "Downloading pre-trained model from $repo_url"
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
git lfs pull --include "data/lang_bpe_500/3gram.pt"
git lfs pull --include "data/lang_bpe_500/4gram.pt"
git lfs pull --include "data/lang_bpe_500/L.pt"
git lfs pull --include "data/lang_bpe_500/LG.pt"
git lfs pull --include "data/lang_bpe_500/Linv.pt"
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "exp/pretrained.pt"
ln -s pretrained.pt epoch-99.pt
ls -lh *.pt
popd

log "Export to torchscript model"
./zipformer_mmi/export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 99 \
  --avg 1 \
  --jit 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.script()"

./zipformer_mmi/jit_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --nn-model-filename $repo/exp/cpu_jit.pt \
  --lang-dir $repo/data/lang_bpe_500 \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

for method in 1best nbest nbest-rescoring-LG nbest-rescoring-3-gram nbest-rescoring-4-gram; do
  log "$method"

  ./zipformer_mmi/pretrained.py \
    --method $method \
    --checkpoint $repo/exp/pretrained.pt \
    --lang-dir $repo/data/lang_bpe_500 \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done


echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p zipformer_mmi/exp
  ln -s $PWD/$repo/exp/pretrained.pt zipformer_mmi/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh zipformer_mmi/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in 1best nbest nbest-rescoring-LG nbest-rescoring-3-gram nbest-rescoring-4-gram; do
    log "Decoding with $method"

    ./zipformer_mmi/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --nbest-scale 1.2 \
      --hp-scale 1.0 \
      --max-duration $max_duration \
      --lang-dir $repo/data/lang_bpe_500 \
      --exp-dir zipformer_mmi/exp
  done

  rm zipformer_mmi/exp/*.pt
fi
