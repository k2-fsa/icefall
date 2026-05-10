#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:/workspace/CosyVoice
# export HF_HOME="/lustre/fsw/general_sa/yuekaiz/.cache/huggingface"
set -eou pipefail

stage=$1
stop_stage=$2


log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 17 ] && [ $stop_stage -ge 17 ]; then
  echo "cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -"
  if [ ! -L "/workspace/slam" ]; then
    cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -
  fi
  log "stage 17: Training Speech2Speech Model, full parameters"
  exp_dir=./qwen_omni/exp_speech2text_first_multi_en_continuation_second_three_s2s
  pretrained_dir=./qwen_omni/exp_speech2text
  ngpu=4

  latest_checkpoint_step=-1
  # Check if exp_dir exists and is a directory
  if [ -d "$exp_dir" ]; then
    # List directories matching checkpoint-* and find the one with the largest step number
    for checkpoint_dir in $(ls -d $exp_dir/checkpoint-*/ 2>/dev/null | sort -V); do
      checkpoint_name=$(basename "$checkpoint_dir") # e.g., checkpoint-1000
      # Extract step number using parameter expansion
      current_step=${checkpoint_name#checkpoint-}
      # Ensure current_step is a number
      if [[ "$current_step" =~ ^[0-9]+$ ]] && [ "$current_step" -gt "$latest_checkpoint_step" ]; then
        latest_checkpoint_step=$current_step
      fi
    done
  fi

  train_cmd_args="--max-duration 200 \
    --enable-musan False \
    --exp-dir $exp_dir \
    --last-stage-model-path $pretrained_dir/checkpoint-58548/pytorch_model.bin \
    --speech-encoder-path-or-name models/large-v2.pt \
    --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
    --on-the-fly-feats True --on-the-fly-speed-perturb False\
    --deepspeed \
    --huggingface-dataset-path-or-name /lustre/fsw/general_sa/yuekaiz/s2s \
    --deepspeed_config ./qwen_omni/ds_config_zero1.json \
    --use-flash-attn True --on-the-fly-feats True \
    --dataset vocalnet_ultrachat_voiceassistant_instruct_s2s --num-epochs 10 \
    --use-lora True --unfreeze-llm True --unfreeze-speech-projector True --enable-speech-output False"

  if [ "$latest_checkpoint_step" -ge 0 ]; then
    log "Continuing training from checkpoint-$latest_checkpoint_step"
    step=$latest_checkpoint_step
    train_cmd_args="$train_cmd_args --pretrained-model-path $exp_dir/checkpoint-${step}/pytorch_model.bin --sampler-state-dict-path $exp_dir/checkpoint-${step}/sampler.pt"
  else
    log "Starting training from scratch as no checkpoint was found in $exp_dir"
    # No pretrained model or sampler state dict needed for the first run
  fi

  torchrun --nproc_per_node $ngpu --nnodes $SLURM_JOB_NUM_NODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --rdzv_id $SLURM_JOBID ./qwen_omni/train.py \
    $train_cmd_args
fi

if [ $stage -le 18 ] && [ $stop_stage -ge 18 ]; then
  echo "cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -"
  # check if the link exists, if not exist, create it
  if [ ! -L "/workspace/slam" ]; then
    cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -
  fi
  log "stage 17: Training Speech2Speech Model, full parameters"
  exp_dir=./qwen_omni/exp_speech2text_first_multi_en_continuation_second_three_s2s_librispeech
  pretrained_dir=./qwen_omni/exp_speech2text
  ngpu=4

  latest_checkpoint_step=-1
  # Check if exp_dir exists and is a directory
  if [ -d "$exp_dir" ]; then
    # List directories matching checkpoint-* and find the one with the largest step number
    for checkpoint_dir in $(ls -d $exp_dir/checkpoint-*/ 2>/dev/null | sort -V); do
      checkpoint_name=$(basename "$checkpoint_dir") # e.g., checkpoint-1000
      # Extract step number using parameter expansion
      current_step=${checkpoint_name#checkpoint-}
      # Ensure current_step is a number
      if [[ "$current_step" =~ ^[0-9]+$ ]] && [ "$current_step" -gt "$latest_checkpoint_step" ]; then
        latest_checkpoint_step=$current_step
      fi
    done
  fi

  train_cmd_args="--max-duration 200 \
    --enable-musan False \
    --exp-dir $exp_dir \
    --last-stage-model-path $pretrained_dir/checkpoint-58548/pytorch_model.bin \
    --speech-encoder-path-or-name models/large-v2.pt \
    --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
    --on-the-fly-feats True --on-the-fly-speed-perturb False\
    --deepspeed \
    --huggingface-dataset-path-or-name /lustre/fsw/general_sa/yuekaiz/s2s \
    --deepspeed_config ./qwen_omni/ds_config_zero1.json \
    --use-flash-attn True --on-the-fly-feats True \
    --dataset vocalnet_ultrachat_voiceassistant_instruct_s2s_librispeech --num-epochs 10 \
    --use-lora True --unfreeze-llm True --unfreeze-speech-projector True --enable-speech-output False"

  if [ "$latest_checkpoint_step" -ge 0 ]; then
    log "Continuing training from checkpoint-$latest_checkpoint_step"
    step=$latest_checkpoint_step
    train_cmd_args="$train_cmd_args --pretrained-model-path $exp_dir/checkpoint-${step}/pytorch_model.bin --sampler-state-dict-path $exp_dir/checkpoint-${step}/sampler.pt"
  else
    log "Starting training from scratch as no checkpoint was found in $exp_dir"
    # No pretrained model or sampler state dict needed for the first run
  fi

  torchrun --nproc_per_node $ngpu --nnodes $SLURM_JOB_NUM_NODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --rdzv_id $SLURM_JOBID ./qwen_omni/train.py \
    $train_cmd_args
fi

if [ $stage -le 19 ] && [ $stop_stage -ge 19 ]; then
  log "stage 19: Training TTS Model"
  exp_dir=./qwen_omni/exp_tts_ultra_chat_voice_assistant
  exp_dir=./qwen_omni/exp_tts_emilia_en_tts_only_template
  exp_dir=./qwen_omni/exp_tts_emilia_en_tts_three_concat
  pretrained_dir=./qwen_omni/exp_speech2text
  ngpu=4

  latest_checkpoint_step=-1
  # Check if exp_dir exists and is a directory
  if [ -d "$exp_dir" ]; then
    # List directories matching checkpoint-* and find the one with the largest step number
    for checkpoint_dir in $(ls -d $exp_dir/checkpoint-*/ 2>/dev/null | sort -V); do
      checkpoint_name=$(basename "$checkpoint_dir") # e.g., checkpoint-1000
      # Extract step number using parameter expansion
      current_step=${checkpoint_name#checkpoint-}
      # Ensure current_step is a number
      if [[ "$current_step" =~ ^[0-9]+$ ]] && [ "$current_step" -gt "$latest_checkpoint_step" ]; then
        latest_checkpoint_step=$current_step
      fi
    done
  fi
  # --dataset ultra_chat_voice_assistant
  train_cmd_args="--batch-size 30 \
    --exp-dir $exp_dir \
    --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
    --enable-speech-input False \
    --deepspeed \
    --dataset  /lustre/fsw/general_sa/yuekaiz/s2s/VoxBox/manifests_emilia_en \
    --deepspeed_config ./qwen_omni/ds_config_zero1.json \
    --use-flash-attn True  \
    --num-epochs 3 \
    --use-lora False --unfreeze-llm False --enable-speech-output True"

  if [ "$latest_checkpoint_step" -ge 0 ]; then
    log "Continuing training from checkpoint-$latest_checkpoint_step"
    step=$latest_checkpoint_step
    train_cmd_args="$train_cmd_args --pretrained-model-path $exp_dir/checkpoint-${step}/pytorch_model.bin --sampler-state-dict-path $exp_dir/checkpoint-${step}/sampler.pt"
  else
    log "Starting training from scratch as no checkpoint was found in $exp_dir"
    # No pretrained model or sampler state dict needed for the first run
  fi

  torchrun --nproc_per_node $ngpu --nnodes $SLURM_JOB_NUM_NODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --rdzv_id $SLURM_JOBID ./qwen_omni/train_tts.py \
    $train_cmd_args
fi


# if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
#   log "stage 20: Training TTS Model"
#   echo "cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -"
#   if [ ! -L "/workspace/slam" ]; then
#     cd /workspace && ln -s /lustre/fsw/general_sa/yuekaiz/s2s slam && cd -
#   fi
#   exp_dir=./qwen_omni/exp_test
#   ngpu=4

#   latest_checkpoint_step=-1
#   # Check if exp_dir exists and is a directory
#   if [ -d "$exp_dir" ]; then
#     # List directories matching checkpoint-* and find the one with the largest step number
#     for checkpoint_dir in $(ls -d $exp_dir/checkpoint-*/ 2>/dev/null | sort -V); do
#       checkpoint_name=$(basename "$checkpoint_dir") # e.g., checkpoint-1000
#       # Extract step number using parameter expansion
#       current_step=${checkpoint_name#checkpoint-}
#       # Ensure current_step is a number
#       if [[ "$current_step" =~ ^[0-9]+$ ]] && [ "$current_step" -gt "$latest_checkpoint_step" ]; then
#         latest_checkpoint_step=$current_step
#       fi
#     done
#   fi

#   train_cmd_args="--max-duration 150 \
#     --enable-musan False \
#     --exp-dir $exp_dir \
#     --speech-encoder-path-or-name models/large-v2.pt \
#     --llm-path-or-name Qwen/Qwen2.5-0.5B-Instruct \
#     --dataset vocalnet_ultrachat_voiceassistant \
#     --manifest-dir data/fbank \
#     --deepspeed \
#     --deepspeed_config ./qwen_omni/ds_config_zero1.json \
#     --use-flash-attn True --on-the-fly-feats True \
#     --use-lora True --unfreeze-llm True --unfreeze-speech-projector True --enable-speech-output True"

#   if [ "$latest_checkpoint_step" -ge 0 ]; then
#     log "Continuing training from checkpoint-$latest_checkpoint_step"
#     step=$latest_checkpoint_step
#     train_cmd_args="$train_cmd_args --pretrained-model-path $exp_dir/checkpoint-${step}/pytorch_model.bin --sampler-state-dict-path $exp_dir/checkpoint-${step}/sampler.pt"
#   else
#     log "Starting training from scratch as no checkpoint was found in $exp_dir"
#     # No pretrained model or sampler state dict needed for the first run
#   fi

#   torchrun --nproc_per_node $ngpu --nnodes $SLURM_JOB_NUM_NODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --rdzv_id $SLURM_JOBID ./qwen_omni/train.py \
#     $train_cmd_args
# fi


# if [ $stage -le 21 ] && [ $stop_stage -ge 21 ]; then
#   log "stage 21: TTS Decoding Test Set"
#   exp_dir=./qwen_omni/exp_tts
#   torchrun --nproc_per_node=2 ./qwen_omni/decode_tts.py \
#     --exp-dir $exp_dir \
#     --speech-encoder-path-or-name models/large-v2.pt  \
#     --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
#     --pretrained-model-path $exp_dir/checkpoint-32001/pytorch_model.bin \
#     --use-flash-attn True \
#     --enable-speech-output True \
#     --token2wav-path /workspace/CosyVoice2-0.5B \
#     --use-lora True
# fi
