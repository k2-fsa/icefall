#!/usr/bin/env bash

set -eou pipefail

# This is the preparation recipe for PromptASR: https://arxiv.org/pdf/2309.07414

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=-1
stop_stage=100
manifest_dir=data/fbank
subset=medium
topk=10000

. shared/parse_options.sh || exit 1

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Download the meta biasing list for LibriSpeech"
    mkdir -p data/context_biasing
    cd data/context_biasing
    git clone https://github.com/facebookresearch/fbai-speech.git
    cd ../..
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Add rare-words for context biasing to the manifest"
    python zipformer_prompt_asr/utils.py \
        --manifest-dir $manifest_dir \
        --subset $subset \
        --top-k $topk

fi
