#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

log "Downloading pre-trained model from $repo_url"
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

log "Display test files"
tree $repo/
ls -lh $repo/test_wavs/*.wav

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/cpu_jit.pt"
git lfs pull --include "exp/pretrained.pt"
git lfs pull --include "exp/encoder_jit_trace.pt"
git lfs pull --include "exp/decoder_jit_trace.pt"
git lfs pull --include "exp/joiner_jit_trace.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
ls -lh *.pt
popd

log "Export to torchscript model"
./pruned_transducer_stateless7_streaming/export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --decode-chunk-len 32 \
  --epoch 99 \
  --avg 1 \
  --jit 1

ls -lh $repo/exp/*.pt

log "Decode with models exported by torch.jit.script()"

./pruned_transducer_stateless7_streaming/jit_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --nn-model-filename $repo/exp/cpu_jit.pt \
  --decode-chunk-len 32 \
  $repo/test_wavs/1089-134686-0001.wav \
  $repo/test_wavs/1221-135766-0001.wav \
  $repo/test_wavs/1221-135766-0002.wav

log "Export to torchscript model by torch.jit.trace()"
./pruned_transducer_stateless7_streaming/jit_trace_export.py \
  --exp-dir $repo/exp \
  --use-averaged-model false \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --decode-chunk-len 32 \
  --epoch 99 \
  --avg 1

log "Decode with models exported by torch.jit.trace()"

./pruned_transducer_stateless7_streaming/jit_trace_pretrained.py \
  --bpe-model $repo/data/lang_bpe_500/bpe.model \
  --encoder-model-filename $repo/exp/encoder_jit_trace.pt \
  --decoder-model-filename $repo/exp/decoder_jit_trace.pt \
  --joiner-model-filename $repo/exp/joiner_jit_trace.pt \
  --decode-chunk-len 32 \
  $repo/test_wavs/1089-134686-0001.wav

for sym in 1 2 3; do
  log "Greedy search with --max-sym-per-frame $sym"

  ./pruned_transducer_stateless7_streaming/pretrained.py \
    --method greedy_search \
    --max-sym-per-frame $sym \
    --checkpoint $repo/exp/pretrained.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --decode-chunk-len 32 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

for method in modified_beam_search beam_search fast_beam_search; do
  log "$method"

  ./pruned_transducer_stateless7_streaming/pretrained.py \
    --method $method \
    --beam-size 4 \
    --checkpoint $repo/exp/pretrained.pt \
    --bpe-model $repo/data/lang_bpe_500/bpe.model \
    --decode-chunk-len 32 \
    $repo/test_wavs/1089-134686-0001.wav \
    $repo/test_wavs/1221-135766-0001.wav \
    $repo/test_wavs/1221-135766-0002.wav
done

echo "GITHUB_EVENT_NAME: ${GITHUB_EVENT_NAME}"
echo "GITHUB_EVENT_LABEL_NAME: ${GITHUB_EVENT_LABEL_NAME}"
if [[ x"${GITHUB_EVENT_NAME}" == x"schedule" || x"${GITHUB_EVENT_LABEL_NAME}" == x"run-decode"  ]]; then
  mkdir -p pruned_transducer_stateless7_streaming/exp
  ln -s $PWD/$repo/exp/pretrained.pt pruned_transducer_stateless7_streaming/exp/epoch-999.pt
  ln -s $PWD/$repo/data/lang_bpe_500 data/

  ls -lh data
  ls -lh pruned_transducer_stateless7_streaming/exp

  log "Decoding test-clean and test-other"

  # use a small value for decoding with CPU
  max_duration=100
  num_decode_stream=200

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "decoding with $method"

    ./pruned_transducer_stateless7_streaming/decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --max-duration $max_duration \
      --decode-chunk-len 32 \
      --exp-dir pruned_transducer_stateless7_streaming/exp
  done

  for method in greedy_search fast_beam_search modified_beam_search; do
    log "Decoding with $method"

    ./pruned_transducer_stateless7_streaming/streaming_decode.py \
      --decoding-method $method \
      --epoch 999 \
      --avg 1 \
      --use-averaged-model 0 \
      --decode-chunk-len 32 \
      --num-decode-streams $num_decode_stream
      --exp-dir pruned_transducer_stateless7_streaming/exp
  done

  rm pruned_transducer_stateless7_streaming/exp/*.pt
fi
