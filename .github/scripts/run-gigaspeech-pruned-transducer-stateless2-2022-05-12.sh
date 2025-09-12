#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/gigaspeech/ASR

repo_url=https://huggingface.co/wgb14/icefall-asr-gigaspeech-pruned-transducer-stateless2

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_NAME}" == x"workflow_dispatch" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p pruned_transducer_stateless2/exp
  ln -s $PWD/$repo/exp/pretrained-iter-3488000-avg-20.pt pruned_transducer_stateless2/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh data/lang_bpe_500
  ls -lh data/fbank
  ls -lh pruned_transducer_stateless2/exp

  pushd data/fbank
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/cuts_DEV.jsonl.gz
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/cuts_TEST.jsonl.gz
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/feats_DEV.lca
  curl -SL -O https://huggingface.co/csukuangfj/giga-dev-dataset-fbank/resolve/main/data/fbank/feats_TEST.lca

  ln -sf cuts_DEV.jsonl.gz gigaspeech_cuts_DEV.jsonl.gz
  ln -sf cuts_TEST.jsonl.gz gigaspeech_cuts_TEST.jsonl.gz
  popd


  log "Decoding dev and test"

  # use a small value for decoding with CPU
  max_duration=100

  # Test only greedy_search to reduce CI running time
  # for method in greedy_search fast_beam_search modified_beam_search; do
  for method in greedy_search; do
    log "Decoding with $method"

    ./pruned_transducer_stateless2/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --max-duration $max_duration \
      --exp-dir pruned_transducer_stateless2/exp
  done

  rm pruned_transducer_stateless2/exp/*.pt
fi
