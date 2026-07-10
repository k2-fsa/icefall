#!/usr/bin/env bash
set -e
export DISABLE_VERSION_CHECK=1

echo "=== Training script started on $(hostname) at $(date) ==="
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export TRANSFORMERS_NO_GIT=1
export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

ASR_BPE_MODEL="data/Europarl-ST/bpe/asr9/bpe.model"
AST_BPE_MODEL="data/Europarl-ST/bpe/ast9/bpe.model"

TRAIN_CUTS_PATHS="data/Europarl-ST/cuts_data/asr_pre_train/merged_asr9_shuffled.train.jsonl.gz"
VALID_CUTS_PATHS="data/Europarl-ST/cuts_data/asr_pre_train/merged_asr9_shuffled.dev.jsonl.gz"

TRAIN_PY="lcma_srt/train.py"

MAX_DURATION=900
NUM_EPOCHS=50
BASE_LR=0.02
START_EPOCH=1
EXP_DIR="exp/europarl"
mkdir -p ${EXP_DIR}
log_path="${EXP_DIR}/run_$(date '+%Y-%m-%d_%H-%M-%S').log"

MANIFEST_DIR="data/fbank"

ASR_NUM_LAYERS="2,2,2,2,2"
ASR_FF_DIM="512,768,1024,1024,1024"
ASR_ENC_DIM="192,256,384,512,384"
ASR_UNMASK_DIM="192,192,256,256,256"
downsampling_factor_asr="1,2,4,8,4"
cnn_module_kernel_asr="1,31,15,15,15"
num_heads_asr="4,4,4,8,8"

ST_NUM_LAYERS="2,2,2,2,2"
ST_FF_DIM="512,512,256,256,256"
ST_ENC_DIM="384,512,256,256,256"
ST_UNMASK_DIM="256,256,256,256,192"
downsampling_factor_st="1,2,4,4,4"
cnn_module_kernel_st="15,31,31,15,15"
num_heads_st="8,8,8,8,8"


causal=0
CHUNK_SIZE="-1"
LEFT_CONTEXT="-1"

TASK_WEIGHT_ASR=1.0
TASK_WEIGHT_ST=0.0
USE_TRANSDUCER=1
USE_CTC_ASR=1
USE_CTC_ST=0
USE_CR_CTC=1
USE_ATT_DEC=0

torchrun --nproc_per_node=4 \
  ${TRAIN_PY} \
  --world-size 4 \
  --num-workers 16 \
  --num-epochs ${NUM_EPOCHS} \
  --start-epoch ${START_EPOCH} \
  --exp-dir ${EXP_DIR} \
  --bpe-model-asr ${ASR_BPE_MODEL} \
  --bpe-model-st  ${AST_BPE_MODEL} \
  --base-lr ${BASE_LR} \
  --max-duration ${MAX_DURATION} \
  --train-cuts-paths ${TRAIN_CUTS_PATHS} \
  --valid-cuts-paths ${VALID_CUTS_PATHS} \
  --utterance-min-duration 0.3 \
  --utterance-max-duration 30.0 \
  --manifest-dir "${MANIFEST_DIR}" \
  --num-encoder-layers-asr ${ASR_NUM_LAYERS} \
  --feedforward-dim-asr ${ASR_FF_DIM} \
  --encoder-dim-asr ${ASR_ENC_DIM} \
  --encoder-unmasked-dim-asr ${ASR_UNMASK_DIM} \
  --num-encoder-layers-st ${ST_NUM_LAYERS} \
  --feedforward-dim-st ${ST_FF_DIM} \
  --encoder-dim-st ${ST_ENC_DIM} \
  --encoder-unmasked-dim-st ${ST_UNMASK_DIM} \
  --chunk-size "${CHUNK_SIZE}" \
  --left-context-frames "${LEFT_CONTEXT}" \
  --use-transducer ${USE_TRANSDUCER} \
  --use-ctc-asr ${USE_CTC_ASR} \
  --use-ctc-st ${USE_CTC_ST} \
  --use-cr-ctc ${USE_CR_CTC} \
  --use-attention-decoder ${USE_ATT_DEC} \
  --task-weight-asr ${TASK_WEIGHT_ASR} \
  --task-weight-st ${TASK_WEIGHT_ST} \
  --prune-range-asr 10 \
  --prune-range-st 10 \
  --enable-spec-aug 0 \
  --use-fp16 1 \
  --causal ${causal} \
  --full-libri 1 \
  --ctc-loss-scale 0.1 \
  --cr-loss-scale 0.05 \
  --num-buckets 100 \
  --downsampling-factor-st ${downsampling_factor_st} \
  --cnn-module-kernel-st ${cnn_module_kernel_st} \
  --num-heads-st ${num_heads_st} \
  --downsampling-factor-asr ${downsampling_factor_asr} \
  --cnn-module-kernel-asr ${cnn_module_kernel_asr} \
  --num-heads-asr ${num_heads_asr} \
  --freeze-asr 0 \
  --freeze-frontend 0 \
  --lr-epochs 6 \
  --warm-step 2000 \
  --output-downsampling-factor-st 1 \
  --decoder-dim-asr 256 \
  --decoder-dim-st 256 \
  --joiner-dim-asr 256 \
  --joiner-dim-st 256 \
  --use-tgt 0 \
  --enable-st 0 \
  --asr-moe 1 \
  --asr-src 1 \
  --ast-moe 0 \
  --ast-tgt 0 \
  --entropy-reg-asr 0.015 \
  --entropy-reg-ast 0.0 \
  --num-experts-asr 8 \
  --num-experts-ast 0 \
  --dump-moe-routing-stats 1 \
  --tgt-langs "en,de,es,fr,it,nl,pl,pt,ro" \
  --srt-langs "en,de,es,fr,it,nl,pl,pt,ro" \
  > ${log_path} 2>&1

echo "=== Training finished at $(date) ==="
