#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
. ../../../tools/activate_python.sh

set -eou pipefail

stage=0
stop_stage=100

model=pruned_transducer_stateless_w2v
world_size=4

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

ft_model=./transducer_unsupervised_finetuning_d2v_v2/unsup_LJft_trial2/best_locked.pt
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train model"
  ./pruned_transducer_stateless_d2v_v2/train.py \
        --wandb False \
        --use-pseudo-labels True \
        --load-unsupfinetuned-model $ft_model \
        --input-strategy AudioSamples \
        --enable-spec-aug False \
        --multi-optim True \
        --start-epoch 1 \
        --world-size 4 \
        --num-epochs 30 \
        --exp-dir ./pruned_transducer_stateless_d2v_v2/d2v-T-LJft \
        --max-duration 150 \
        --freeze-finetune-updates 100000 \
        --encoder-dim 768 \
        --decoder-dim 768 \
        --joiner-dim 768 \
        --use-fp16 1 \
        --accum-grads 16 \
        --encoder-type d2v \
        --additional-block True \
        --prune-range 10 \
        --context-size 2 \
        --ctc-loss-scale 0.2 \
        --peak-dec-lr 0.04175 \
        --peak-enc-lr 0.0003859 \
        --update-ema False \
        --layer-average False
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decoding"
  # modified_beam_search, greedy_search, ctc_greedy_search
  expdir=./pruned_transducer_stateless_d2v_v2/d2v-T-LJft-oracle-epoch100
  for method in ctc_greedy_search; do
    ./pruned_transducer_stateless_d2v_v2/decode.py \
      --gen-pseudo-label False \
      --input-strategy AudioSamples \
      --enable-spec-aug False \
      --additional-block True \
      --model-name best-valid-loss.pt \
      --exp-dir $expdir \
      --max-duration 400 \
      --decoding-method $method \
      --max-sym-per-frame 1 \
      --encoder-type d2v \
      --encoder-dim 768 \
      --decoder-dim 768 \
      --joiner-dim 768
  done
fi