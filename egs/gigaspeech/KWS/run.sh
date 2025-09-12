#!/usr/bin/env bash

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONPATH=../../../:$PYTHONPATH

stage=0
stop_stage=100

. shared/parse_options.sh || exit 1

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Train a model."
  if [ ! -e data/fbank/.gigaspeech.done ]; then
    log "You need to run the prepare.sh first."
    exit -1
  fi
  
  python ./zipformer/train.py \
      --world-size 4 \
      --exp-dir zipformer/exp \
      --decoder-dim 320 \
      --joiner-dim 320 \
      --num-encoder-layers 1,1,1,1,1,1 \
      --feedforward-dim 192,192,192,192,192,192 \
      --encoder-dim 128,128,128,128,128,128 \
      --encoder-unmasked-dim 128,128,128,128,128,128 \
      --num-epochs 12 \
      --lr-epochs 1.5 \
      --use-fp16 1 \
      --start-epoch 1 \
      --subset XL \
      --bpe-model data/lang_bpe_500/bpe.model \
      --causal 1 \
      --max-duration 1000
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Decode the model."

  export CUDA_VISIBLE_DEVICES="0"
  for t in small large; do
    python ./zipformer/decode.py \
        --epoch 12 \
        --avg 2 \
        --exp-dir ./zipformer/exp \
        --bpe-model data/lang_bpe_500/bpe.model \
        --causal 1 \
        --chunk-size 16 \
        --left-context-frames 64 \
        --decoder-dim 320 \
        --joiner-dim 320 \
        --num-encoder-layers 1,1,1,1,1,1 \
        --feedforward-dim 192,192,192,192,192,192 \
        --encoder-dim 128,128,128,128,128,128 \
        --encoder-unmasked-dim 128,128,128,128,128,128 \
        --test-set $t \
        --keywords-score 1.0 \
        --keywords-threshold 0.35 \
        --keywords-file ./data/commands_${t}.txt  \
        --max-duration 3000
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Export the model."

  python ./zipformer/export.py \
      --epoch 12 \
      --avg 2 \
      --exp-dir ./zipformer/exp \
      --tokens data/lang_bpe_500/tokens.txt \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 64 \
      --decoder-dim 320 \
      --joiner-dim 320 \
      --num-encoder-layers 1,1,1,1,1,1 \
      --feedforward-dim 192,192,192,192,192,192 \
      --encoder-dim 128,128,128,128,128,128 \
      --encoder-unmasked-dim 128,128,128,128,128,128

  python ./zipformer/export-onnx-streaming.py \
    --exp-dir zipformer/exp \
    --tokens data/lang_bpe_500/tokens.txt \
    --epoch 12 \
    --avg 2 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers 1,1,1,1,1,1 \
    --feedforward-dim 192,192,192,192,192,192 \
    --encoder-dim 128,128,128,128,128,128 \
    --encoder-unmasked-dim 128,128,128,128,128,128 \
    --causal 1
fi 

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 2: Finetune the model"
  
  # The following configuration of lr schedule should work well
  # You may also tune the following parameters to adjust learning rate schedule
  base_lr=0.0005
  lr_epochs=100
  lr_batches=100000

  # We recommend to start from an averaged model
  finetune_ckpt=zipformer/exp/pretrained.pt

  ./zipformer/finetune.py \
    --world-size 4 \
    --num-epochs 10 \
    --start-epoch 1 \
    --exp-dir zipformer/exp_finetune \
    --bpe-model data/lang_bpe_500/bpe.model \
    --use-fp16 1 \
    --use-mux 1 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers 1,1,1,1,1,1 \
    --feedforward-dim 192,192,192,192,192,192 \
    --encoder-dim 128,128,128,128,128,128 \
    --encoder-unmasked-dim 128,128,128,128,128,128 \
    --causal 1 \
    --base-lr $base_lr \
    --lr-epochs $lr_epochs \
    --lr-batches $lr_batches \
    --finetune-ckpt $finetune_ckpt \
    --max-duration 1500
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 1: Decode the finetuned model."
  export CUDA_VISIBLE_DEVICES="0"
  for t in small large; do
    python ./zipformer/decode.py \
        --epoch 10 \
        --avg 2 \
        --exp-dir ./zipformer/exp_finetune \
        --bpe-model data/lang_bpe_500/bpe.model \
        --causal 1 \
        --chunk-size 16 \
        --left-context-frames 64 \
        --decoder-dim 320 \
        --joiner-dim 320 \
        --num-encoder-layers 1,1,1,1,1,1 \
        --feedforward-dim 192,192,192,192,192,192 \
        --encoder-dim 128,128,128,128,128,128 \
        --encoder-unmasked-dim 128,128,128,128,128,128 \
        --test-set $t \
        --keywords-score 1.0 \
        --keywords-threshold 0.35 \
        --keywords-file ./data/commands_${t}.txt  \
        --max-duration 3000
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 2: Export the finetuned model."

  python ./zipformer/export.py \
      --epoch 10 \
      --avg 2 \
      --exp-dir ./zipformer/exp_finetune \
      --tokens data/lang_bpe_500/tokens.txt \
      --causal 1 \
      --chunk-size 16 \
      --left-context-frames 64 \
      --decoder-dim 320 \
      --joiner-dim 320 \
      --num-encoder-layers 1,1,1,1,1,1 \
      --feedforward-dim 192,192,192,192,192,192 \
      --encoder-dim 128,128,128,128,128,128 \
      --encoder-unmasked-dim 128,128,128,128,128,128

  python ./zipformer/export-onnx-streaming.py \
    --exp-dir zipformer/exp_finetune \
    --tokens data/lang_bpe_500/tokens.txt \
    --epoch 10 \
    --avg 2 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --decoder-dim 320 \
    --joiner-dim 320 \
    --num-encoder-layers 1,1,1,1,1,1 \
    --feedforward-dim 192,192,192,192,192,192 \
    --encoder-dim 128,128,128,128,128,128 \
    --encoder-unmasked-dim 128,128,128,128,128,128 \
    --causal 1
fi 
