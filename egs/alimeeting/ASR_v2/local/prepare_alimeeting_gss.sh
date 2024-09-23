#!/bin/bash
# This script is used to run GSS-based enhancement on AMI data.
set -euo pipefail
nj=4
stage=0

. shared/parse_options.sh || exit 1

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/prepare_alimeeting_gss.sh [options] <data-dir> <exp-dir>"
   echo "e.g. local/prepare_alimeeting_gss.sh data/manifests exp/ami_gss"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --stage <stage>                          # stage to start running from"
   exit 1;
fi

DATA_DIR=$1
EXP_DIR=$2

mkdir -p $EXP_DIR

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 1 ]; then
  log "Stage 1: Prepare cut sets"
  for part in train eval test; do
    lhotse cut simple \
      -r $DATA_DIR/alimeeting-mdm_recordings_${part}.jsonl.gz \
      -s $DATA_DIR/alimeeting-mdm_supervisions_${part}.jsonl.gz \
      $EXP_DIR/cuts_${part}.jsonl.gz
  done
fi

if [ $stage -le 2 ]; then
  log "Stage 2: Trim cuts to supervisions (1 cut per supervision segment)"
  for part in train eval test; do
    lhotse cut trim-to-supervisions --discard-overlapping \
        $EXP_DIR/cuts_${part}.jsonl.gz $EXP_DIR/cuts_per_segment_${part}.jsonl.gz
  done
fi

if [ $stage -le 3 ]; then
  log "Stage 3: Split manifests for multi-GPU processing (optional)"
  for part in train eval test; do
    gss utils split $nj $EXP_DIR/cuts_per_segment_${part}.jsonl.gz \
      $EXP_DIR/cuts_per_segment_${part}_split$nj
  done
fi

if [ $stage -le 4 ]; then
  log "Stage 4: Enhance train segments using GSS (requires GPU)"
  # for train, we use smaller context and larger batches to speed-up processing
  for JOB in $(seq $nj); do
    gss enhance cuts $EXP_DIR/cuts_train.jsonl.gz \
      $EXP_DIR/cuts_per_segment_train_split$nj/cuts_per_segment_train.$JOB.jsonl.gz $EXP_DIR/enhanced \
      --bss-iterations 10 \
      --context-duration 5.0 \
      --use-garbage-class \
      --channels 0,1,2,3,4,5,6,7 \
      --min-segment-length 0.05 \
      --max-segment-length 25.0 \
      --max-batch-duration 60.0 \
      --num-buckets 4 \
      --num-workers 4
  done
fi

if [ $stage -le 5 ]; then
  log "Stage 5: Enhance eval/test segments using GSS (using GPU)"
  # for eval/test, we use larger context and smaller batches to get better quality
  for part in eval test; do
    for JOB in $(seq $nj); do
      gss enhance cuts $EXP_DIR/cuts_${part}.jsonl.gz \
      $EXP_DIR/cuts_per_segment_${part}_split$nj/cuts_per_segment_${part}.$JOB.jsonl.gz \
      $EXP_DIR/enhanced \
      --bss-iterations 10 \
      --context-duration 15.0 \
      --use-garbage-class \
      --channels 0,1,2,3,4,5,6,7 \
      --min-segment-length 0.05 \
      --max-segment-length 16.0 \
      --max-batch-duration 45.0 \
      --num-buckets 4 \
      --num-workers 4
    done
  done
fi

if [ $stage -le 6 ]; then
  log "Stage 6: Prepare manifests for GSS-enhanced data"
  python local/prepare_alimeeting_enhanced.py $DATA_DIR $EXP_DIR/enhanced -j $nj --min-segment-duration 0.05
fi
