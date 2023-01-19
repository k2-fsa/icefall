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

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train model"
  ./transducer_unsupervised_finetuning_d2v_v2/train.py \
        --wandb False \
        --use-pseudo-labels True \
        --load-prefinetuned-model ./model_locked/libri_finetuned.pt \
        --input-strategy AudioSamples \
        --enable-spec-aug False \
        --multi-optim True \
        --start-epoch 1 \
        --world-size 4 \
        --num-epochs 60 \
        --exp-dir ./transducer_unsupervised_finetuning_d2v_v2/d2v-T-LJft \
        --max-duration 150 \
        --freeze-finetune-updates 1000 \
        --encoder-dim 768 \
        --decoder-dim 768 \
        --joiner-dim 768 \
        --use-fp16 1 \
        --accum-grads 16 \
        --encoder-type d2v \
        --additional-block True \
        --peak-dec-lr 0.04175 \
        --peak-enc-lr 0.0003859 \
        --update-ema True \
        --layer-average False
fi