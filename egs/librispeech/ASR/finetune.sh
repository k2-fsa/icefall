#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

stage=-1
stop_stage=100

# This is an example script for fine-tuning. Here, we fine-tune a model trained
# on Librispeech on GigaSpeech. The model used for fine-tuning is 
# pruned_transducer_stateless7 (zipformer). If you want to fine-tune model 
# from another recipe, you can adapt ./pruned_transducer_stateless7/finetune.py 
# for that recipe. If you have any problem, please open up an issue in https://github.com/k2-fsa/icefall/issues.

# We assume that you have already prepared the GigaSpeech manfiest&features under ./data.
# If you haven't done that, please see https://github.com/k2-fsa/icefall/blob/master/egs/gigaspeech/ASR/prepare.sh.

dl_dir=$PWD/download

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
  git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Start fine-tuning"
  
  # The following configuration of lr schedule should work well
  # You may also tune the following parameters to adjust learning rate schedule
  base_lr=0.005
  lr_epochs=100
  lr_batches=100000

  # We recommend to start from an averaged model
  finetune_ckpt=icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained.pt
  export CUDA_VISIBLE_DEVICES="0,1"

  ./pruned_transducer_stateless7/finetune.py \
    --world-size 2 \
    --master-port 18180 \
    --num-epochs 20 \
    --start-epoch 1 \
    --exp-dir pruned_transducer_stateless7/exp_giga_finetune \
    --subset S \
    --use-fp16 1 \
    --base-lr $base_lr \
    --lr-epochs $lr_epochs \
    --lr-batches $lr_batches \
    --bpe-model icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/data/lang_bpe_500/bpe.model \
    --do-finetune True \
    --finetune-ckpt $finetune_ckpt \
    --max-duration 500
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"

  epoch=15
  avg=10

  for m in greedy_search modified_beam_search; do
    python pruned_transducer_stateless7/decode_gigaspeech.py \
    --epoch $epoch \
    --avg $avg \
    --use-averaged-model True \
    --beam-size 4 \
    --exp-dir pruned_transducer_stateless7/exp_giga_finetune \
    --bpe-model icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/data/lang_bpe_500/bpe.model \
    --max-duration 400 \
    --decoding-method $m
  done
fi
