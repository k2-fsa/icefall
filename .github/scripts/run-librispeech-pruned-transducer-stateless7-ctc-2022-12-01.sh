#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01

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
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "exp/pretrained.pt"
ln -s pretrained.pt epoch-99.pt
ls -lh *.pt
popd

log "Export to torchscript model"
./pruned_transducer_stateless7_ctc/export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 99 \
  --avg 1 \
  --jit 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.script()"

./pruned_transducer_stateless7_ctc/jit_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --nn-model-filename $repo/exp/cpu_jit.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

for m in ctc-decoding 1best; do
  ./pruned_transducer_stateless7_ctc/jit_pretrained_ctc.py \
    --model-filename $repo/exp/cpu_jit.pt \
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

for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./pruned_transducer_stateless7_ctc/pretrained.py \
    --method greedy_search \
    --max-sym-per-frame $sym \
    --checkpoint $repo/exp/pretrained.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./pruned_transducer_stateless7_ctc/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

for m in ctc-decoding 1best; do
  ./pruned_transducer_stateless7_ctc/pretrained_ctc.py \
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
  mkdir -p pruned_transducer_stateless7_ctc/exp
  ln -s $PWD/$repo/exp/pretrained.pt pruned_transducer_stateless7_ctc/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh pruned_transducer_stateless7_ctc/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "Decoding with $method"

    ./pruned_transducer_stateless7_ctc/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --exp-dir pruned_transducer_stateless7_ctc/exp
  done

  for m in ctc-decoding 1best; do
    ./pruned_transducer_stateless7_ctc/ctc_decode.py \
        --epoch 999 \
        --avg 1 \
        --exp-dir ./pruned_transducer_stateless7_ctc/exp \
        --max-duration $max_duration \
        --use-averaged-model 0 \
        --decoding-method $m \
        --hlg-scale 0.6 \
        --lm-dir data/lm
  done

  rm pruned_transducer_stateless7_ctc/exp/*.pt
fi
