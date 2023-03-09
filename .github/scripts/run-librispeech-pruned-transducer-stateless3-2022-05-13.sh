#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

log "Downloading pre-trained model from $repo_url"
git lfs install
git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo/exp
ln -s pretrained-iter-1224000-avg-14.pt pretrained.pt
ln -s pretrained-iter-1224000-avg-14.pt epoch-99.pt
popd


log "Export to torchscript model"
./pruned_transducer_stateless3/export.py \
  --exp-dir $repo/exp \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 99 \
  --avg 1 \
  --jit 1

./pruned_transducer_stateless3/export.py \
  --exp-dir $repo/exp \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --epoch 99 \
  --avg 1 \
  --jit-trace 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.trace()"

./pruned_transducer_stateless3/jit_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
  --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
  --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Decode with models exported by torch.jit.script()"

./pruned_transducer_stateless3/jit_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --encoder-model-filename $repo/exp/encoder_jit_script.pt \
  --decoder-model-filename $repo/exp/decoder_jit_script.pt \
  --joiner-model-filename $repo/exp/joiner_jit_script.pt \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav


for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./pruned_transducer_stateless3/pretrained.py \
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

  ./pruned_transducer_stateless3/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p pruned_transducer_stateless3/exp
  ln -s $PWD/$repo/exp/pretrained.pt pruned_transducer_stateless3/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh pruned_transducer_stateless3/exp

  log "Decoding test-clean and test-other"

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
