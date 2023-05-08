#!/usr/bin/env bash

# Please run ./prepare.sh first

stage=-1
stop_stage=100

# Number of GPUs to use for training
world_size=1

# Number of epochs to train
num_epochs=20

# Use this epoch for computing ppl
use_epoch=19

# number of models to average for computing ppl
use_avg=2

exp_dir=./my-rnnlm-exp

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Training RNN LM"

  ./rnn_lm/train.py \
    --exp-dir $exp_dir \
    --start-epoch 0 \
    --num-epochs $num_epochs \
    --world-size $world_size \
    --use-fp16 0 \
    --vocab-size 500 \
    \
    --lm-data ./data/lm_training_bpe_500/sorted_lm_data.pt \
    --lm-data-valid ./data/lm_training_bpe_500/sorted_lm_data-valid.pt \
    \
    --embedding-dim 800 \
    --hidden-dim 200 \
    --num-layers 2 \
    --tie-weights false \
    --batch-size 50
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Computing perplexity"

  ./rnn_lm/compute_perplexity.py \
    --exp-dir $exp_dir \
    --epoch $use_epoch \
    --avg $use_avg \
    --vocab-size 500 \
    \
    --lm-data ./data/lm_training_bpe_500/sorted_lm_data-test.pt \
    \
    --embedding-dim 800 \
    --hidden-dim 200 \
    --num-layers 2 \
    --tie-weights false \
    --batch-size 50
fi
