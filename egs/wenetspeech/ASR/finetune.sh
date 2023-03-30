#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

# This is an example script for fine-tuning. Here, we fine-tune a model trained
# on WenetSpeech on Aishell. The model used for fine-tuning is
# pruned_transducer_stateless2 (zipformer). If you want to fine-tune model
# from another recipe, you can adapt ./pruned_transducer_stateless2/finetune.py
# for that recipe. If you have any problem, please open up an issue in https://github.com/k2-fsa/icefall/issues.

# We assume that you have already prepared the Aishell manfiest&features under ./data.
# If you haven't done that, please see https://github.com/k2-fsa/icefall/blob/master/egs/aishell/ASR/prepare.sh.

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "Stage -1: Download Pre-trained model"

  # clone from huggingface
  git lfs install
  git clone https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2

fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Start fine-tuning"

  # The following configuration of lr schedule should work well
  # You may also tune the following parameters to adjust learning rate schedule
  initial_lr=0.0001
  lr_epochs=100
  lr_batches=100000

  # We recommend to start from an averaged model
  finetune_ckpt=icefall_asr_wenetspeech_pruned_transducer_stateless2/exp/pretrained_epoch_10_avg_2.pt
  lang_dir=icefall_asr_wenetspeech_pruned_transducer_stateless2/data/lang_char
  export CUDA_VISIBLE_DEVICES="0,1"

  ./pruned_transducer_stateless2/finetune.py \
    --world-size 2 \
    --master-port 18180 \
    --num-epochs 15 \
    --context-size 2 \
    --exp-dir pruned_transducer_stateless2/exp_aishell_finetune \
    --initial-lr $initial_lr \
    --lr-epochs $lr_epochs \
    --lr-batches $lr_batches \
    --lang-dir $lang_dir \
    --do-finetune True \
    --finetune-ckpt $finetune_ckpt \
    --max-duration 200
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"

  epoch=4
  avg=4

  for m in greedy_search modified_beam_search; do
    python pruned_transducer_stateless2/decode_aishell.py \
    --epoch $epoch \
    --avg $avg \
    --context-size 2 \
    --beam-size 4 \
    --exp-dir pruned_transducer_stateless2/exp_aishell_finetune \
    --max-duration 400 \
    --decoding-method $m
  done
fi
