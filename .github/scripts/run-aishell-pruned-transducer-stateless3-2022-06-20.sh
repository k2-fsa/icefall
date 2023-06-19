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

log "Downloading pre-trained model from $repo_url"
repo_url=https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
ln -s pretrained-epoch-29-avg-5-torch-1.10.0.pt pretrained.pt
popd

for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./pruned_transducer_stateless3/pretrained.py \
    --method greedy_search \
    --max-sym-per-frame $sym \
    --checkpoint $repo/exp/pretrained.pt \
    --lang-dir $repo/data/lang_char \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./pruned_transducer_stateless3/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --lang-dir $repo/data/lang_char \
    $repo/test_wavs/BAC009S0764W0121.wav \
    $repo/test_wavs/BAC009S0764W0122.wav \
    $repo/test_wavs/BAC009S0764W0123.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p pruned_transducer_stateless3/exp
  ln -s $PWD/$repo/exp/pretrained.pt pruned_transducer_stateless3/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_char data/

  ls -lh data
  ls -lh pruned_transducer_stateless3/exp

  log "Decoding test and dev"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "Decoding with $method"

    ./pruned_transducer_stateless3/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --max-duration $max_duration \
      --exp-dir pruned_transducer_stateless3/exp
  done

  rm pruned_transducer_stateless3/exp/*.pt
fi
