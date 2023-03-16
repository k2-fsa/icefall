#!/usr/bin/env bash

set -eou pipefail

# You need to execute ./prepare.sh to prepare datasets.
stage=0
stop_stage=2

epoch=10
avg=1
max_duration=150
exp_dir=./ctc_tdnn/exp_max_duration_${max_duration}/
epoch_avg=epoch_${epoch}-avg_${avg}
post_dir=${exp_dir}/post/${epoch_avg}

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Model training"
  python ./ctc_tdnn/train.py \
    --num-epochs $epoch \
    --exp-dir $exp_dir
    --max-duration $max_duration
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Get posterior of test sets"
  python ctc_tdnn/inference.py \
    --avg $avg \
    --epoch $epoch \
    --exp-dir $exp_dir
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Decode and compute area under curve(AUC)"
  for test_set in test aishell_test cw_test; do
    python ctc_tdnn/decode.py \
      --post-h5 ${post_dir}/${test_set}.h5 \
      --score-file ${post_dir}/fst_${test_set}_score.txt
  done
  python ./local/auc.py   \
    --legend himia_cw \
    --positive-score-file ${post_dir}/fst_test_score.txt \
    --negative-score-file ${post_dir}/fst_cw_test_score.txt

  python ./local/auc.py \
    --legend himia_aishell \
    --positive-score-file ${post_dir}/fst_test_score.txt \
    --negative-score-file ${post_dir}/fst_aishell_test_score.txt
fi
