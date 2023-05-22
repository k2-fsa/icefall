#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

export CUDA_VISIBLE_DEVICES="0,1,2,3"

set -eou pipefail

# This script is used to recogize long audios. The process is as follows:
# 1) Split long audios into chunks with overlaps.
# 2) Perform speech recognition on chunks, getting tokens and timestamps.
# 3) Merge the overlapped chunks into utterances acording to the timestamps.

# Each chunk (except the first and the last) is padded with extra left side and right side.
# The chunk length is: left_side + chunk_size + right_side.
chunk=30.0
extra=2.0

stage=1
stop_stage=4

# We assume that you have downloaded the LibriLight dataset
# with audio files in $corpus_dir and texts in $text_dir
corpus_dir=$PWD/download/libri-light
text_dir=$PWD/download/librilight_text
# Path to save the manifests
output_dir=$PWD/data/librilight

world_size=4


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # We will get librilight_recodings_{subset}.jsonl.gz and librilight_supervisions_{subset}.jsonl.gz
  # saved in $output_dir/manifests
  log "Stage 1: Prepare LibriLight manifest"
  lhotse prepare librilight $corpus_dir $text_dir $output_dir/manifests -j 10
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # Chunk manifests are saved to $output_dir/manifests_chunk/librilight_cuts_{subset}.jsonl.gz
  log "Stage 2: Split long audio into chunks"
  ./long_file_recog/split_into_chunks.py \
    --manifest-in-dir $output_dir/manifests \
    --manifest-out-dir $output_dir/manifests_chunk \
    --chunk $chunk \
    --extra $extra  # Extra duration (in seconds) at both sides
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  # Recognized tokens and timestamps are saved to $output_dir/manifests_chunk_recog/librilight_cuts_{subset}.jsonl.gz

  # This script loads torchscript models, exported by `torch.jit.script()`,
  # and uses it to decode waves.
  # You can download the jit model from https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

  log "Stage 3: Perform speech recognition on splitted chunks"
  for subset in small median large; do
    ./long_file_recog/recognize.py \
      --world-size $world_size \
      --num-workers 8 \
      --subset $subset \
      --manifest-in-dir $output_dir/manifests_chunk \
      --manifest-out-dir $output_dir/manifests_chunk_recog \
      --nn-model-filename long_file_recog/exp/jit_model.pt \
      --bpe-model data/lang_bpe_500/bpe.model \
      --max-duration 2400 \
      --decoding-method greedy_search
      --master 12345

    if [ $world_size -gt 1 ]; then
      # Combine manifests from different jobs
      lhotse combine $(find $output_dir/manifests_chunk_recog -name librilight_cuts_${subset}_job_*.jsonl.gz | tr "\n" " ") $output_dir/manifests_chunk_recog/librilight_cuts_${subset}.jsonl.gz
    fi
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  # Final results are saved in $output_dir/manifests/librilight_cuts_{subset}.jsonl.gz
  log "Stage 4: Merge splitted chunks into utterances."
  ./long_file_recog/merge_chunks.py \
    --manifest-in-dir $output_dir/manifests_chunk_recog \
    --manifest-out-dir $output_dir/manifests \
    --bpe-model data/lang_bpe_500/bpe.model \
    --extra $extra
fi


