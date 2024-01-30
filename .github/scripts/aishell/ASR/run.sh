#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/aishell/ASR

function download_test_dev_manifests() {
  git lfs install

  fbank_url=https://huggingface.co/csukuangfj/aishell-test-dev-manifests
  log "Downloading pre-commputed fbank from $fbank_url"

  git clone https://huggingface.co/csukuangfj/aishell-test-dev-manifests
  ln -s $PWD/aishell-test-dev-manifests/data .
}

function test_transducer_stateless3_2022_06_20() {
  repo_url=https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20
  log "Downloading pre-trained model from $repo_url"
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  pushd $repo/exp
  ln -s pretrained-epoch-29-avg-5-torch-1.10.0.pt pretrained.pt
  popd

  log "test greedy_search with pretrained.py"

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

  log "test beam search with pretrained.py"

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

  rm -rf $repo
}

function test_zipformer_large_2023_10_24() {
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
  rm -rf $repo
}

function test_zipformer_2023_10_24() {
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
  rm -rf $repo
}

function test_zipformer_small_2023_10_24() {
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
  rm -rf $repo
}

function test_transducer_stateless_modified_2022_03_01() {
  repo_url=https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2022-03-01

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless_modified/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --lang-dir $repo/data/lang_char \
      $repo/test_wavs/BAC009S0764W0121.wav \
      $repo/test_wavs/BAC009S0764W0122.wav \
      $repo/test_wavs/BAC009S0764W0123.wav
  done

  for method in modified_beam_search beam_search; do
    log "$method"

    ./transducer_stateless_modified/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --lang-dir $repo/data/lang_char \
      $repo/test_wavs/BAC009S0764W0121.wav \
      $repo/test_wavs/BAC009S0764W0122.wav \
      $repo/test_wavs/BAC009S0764W0123.wav
  done
  rm -rf $repo
}

function test_transducer_stateless_modified_2_2022_03_01() {
  repo_url=https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2-2022-03-01

  log "Downloading pre-trained model from $repo_url"
  git lfs install
  git clone $repo_url
  repo=$(basename $repo_url)

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  for sym in 1 2 3; do
    log "Greedy search with --max-sym-per-frame $sym"

    ./transducer_stateless_modified-2/pretrained.py \
      --method greedy_search \
      --max-sym-per-frame $sym \
      --checkpoint $repo/exp/pretrained.pt \
      --lang-dir $repo/data/lang_char \
      $repo/test_wavs/BAC009S0764W0121.wav \
      $repo/test_wavs/BAC009S0764W0122.wav \
      $repo/test_wavs/BAC009S0764W0123.wav
  done

  for method in modified_beam_search beam_search; do
    log "$method"

    ./transducer_stateless_modified-2/pretrained.py \
      --method $method \
      --beam-size 4 \
      --checkpoint $repo/exp/pretrained.pt \
      --lang-dir $repo/data/lang_char \
      $repo/test_wavs/BAC009S0764W0121.wav \
      $repo/test_wavs/BAC009S0764W0122.wav \
      $repo/test_wavs/BAC009S0764W0123.wav
  done
  rm -rf $repo
}

function test_conformer_ctc() {
  repo_url=https://huggingface.co/csukuangfj/icefall_asr_aishell_conformer_ctc
  log "Downloading pre-trained model from $repo_url"
  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  repo=$(basename $repo_url)
  pushd $repo

  git lfs pull --include "exp/pretrained.pt"
  git lfs pull --include "data/lang_char/H.fst"
  git lfs pull --include "data/lang_char/HL.fst"
  git lfs pull --include "data/lang_char/HLG.fst"

  popd

  log "Display test files"
  tree $repo/
  ls -lh $repo/test_wavs/*.wav

  log "CTC decoding"

  log "Exporting model with torchscript"

  pushd $repo/exp
  ln -s pretrained.pt epoch-99.pt
  popd

  ./conformer_ctc/export.py \
    --epoch 99 \
    --avg 1 \
    --exp-dir $repo/exp \
    --tokens $repo/data/lang_char/tokens.txt \
    --jit 1

  ls -lh $repo/exp

  ls -lh $repo/data/lang_char

  log "Decoding with H on CPU with OpenFst"

  ./conformer_ctc/jit_pretrained_decode_with_H.py \
    --nn-model $repo/exp/cpu_jit.pt \
    --H $repo/data/lang_char/H.fst \
    --tokens $repo/data/lang_char/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "Decoding with HL on CPU with OpenFst"

  ./conformer_ctc/jit_pretrained_decode_with_HL.py \
    --nn-model $repo/exp/cpu_jit.pt \
    --HL $repo/data/lang_char/HL.fst \
    --words $repo/data/lang_char/words.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "Decoding with HLG on CPU with OpenFst"

  ./conformer_ctc/jit_pretrained_decode_with_HLG.py \
    --nn-model $repo/exp/cpu_jit.pt \
    --HLG $repo/data/lang_char/HLG.fst \
    --words $repo/data/lang_char/words.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  rm -rf $repo
}

download_test_dev_manifests
test_transducer_stateless3_2022_06_20
test_zipformer_large_2023_10_24
test_zipformer_2023_10_24
test_zipformer_small_2023_10_24
test_transducer_stateless_modified_2022_03_01
test_transducer_stateless_modified_2_2022_03_01
# test_conformer_ctc # fails for torch 1.13.x and torch 2.0.x
