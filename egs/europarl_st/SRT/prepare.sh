#!/usr/bin/env bash

set -eou pipefail

# This script prepares the Europarl-ST dataset for joint multilingual
# speech recognition and translation (SRT) training.
#
# Usage:
#   cd egs/europarl_st/SRT
#   bash prepare.sh --stage 0 --stop-stage 5
#
# Prerequisites:
#   - Download Europarl-ST v1.1 from https://www.mllp.upv.es/europarl-st/
#   - Place it under data/europarl_st/v1.1/

stage=0
stop_stage=5

# Paths (modify as needed)
data_root=./data/europarl_st
raw_dir=${data_root}/v1.1
audio_dir=${data_root}/audio
texts_dir=${data_root}/texts
norm_dir=${data_root}/normalizer
manifest_dir=${data_root}/manifests
fbank_dir=${data_root}/fbank
bpe_dir=${data_root}/bpe

. shared/parse_options.sh || true

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Extract audio segments and build JSONL"
  python local/org_to_jsonl.py \
    --data-dir ${raw_dir} \
    --output-dir ${audio_dir}
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Normalize text"
  python local/normalize_texts.py \
    --src-dir ${texts_dir} \
    --dst-dir ${norm_dir} \
    --fields text st_text \
    --normalizer basic \
    --skip-existing
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Generate CutSet manifests with FBANK features"
  python local/texts_to_cuts.py \
    --src-dir ${norm_dir} \
    --dst-dir ${manifest_dir} \
    --audio-root ${data_root} \
    --storage-root ${fbank_dir} \
    --num-workers 8 \
    --skip-missing-audio
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Filter out entries with empty text"
  python local/filter_cuts_texts.py \
    --manifest-dir ${manifest_dir} \
    --overwrite
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Validate manifests"
  python local/check_manifests.py \
    --manifests-dir ${manifest_dir} \
    --num-workers 8
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Train BPE models"

  # ASR BPE (9-language shared)
  mkdir -p ${bpe_dir}/asr9
  python local/train_bpe.py \
    --lang-dir ${bpe_dir}/asr9 \
    --transcript ${data_root}/asr_train_text.txt \
    --vocab-size 500

  # ST BPE (9-language shared)
  mkdir -p ${bpe_dir}/ast9
  python local/train_bpe.py \
    --lang-dir ${bpe_dir}/ast9 \
    --transcript ${data_root}/ast_train_text.txt \
    --vocab-size 6000
fi

log "Data preparation completed successfully!"
