#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/marcoyang/icefall-asr-librispeech-finetune-hubert-transducer-2022-12-26

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
soxi $repo/test_wavs/*.wav
ls -lh $repo/test_wavs/*.wav

pushd $repo/train_960h_hubert_large
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "train_960h_hubert_large/epoch-9999.pt"
ls -lh *.pt
popd

# download un-fintuned hubert large model from fairseq
./finetune_hubert_transducer/download.sh large

for method in greedy_search modified_beam_search; do
  log "$method"

  ./finetune_hubert_transducer/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/train_960h_hubert_large/epoch-9999.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --hubert-model-dir ./finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt \
    --hubert-subsample-output 1 \
    --hubert-subsample-mode concat_tanh \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode" || x"${GITHUB_EVENT_LABEL_NAME}" == x"hubert"  ]]; then
  mkdir -p finetune_hubert_transducer/exp
  ln -s $PWD/$repo/train_960h_hubert_large/epoch-9999.pt finetune_hubert_transducer/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh finetune_hubert_transducer/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "Decoding with $method"

    ./finetune_hubert_transducer/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --exp-dir finetune_hubert_transducer/exp \
      --hubert-model-dir ./finetune_hubert_transducer/pretrained_models/hubert_large_ll60k.pt \
      --hubert-subsample-output 1 \
      --hubert-subsample-mode concat_tanh \
      --input-strategy AudioSamples
  done

  rm finetune_hubert_transducer/exp/*.pt
fi
