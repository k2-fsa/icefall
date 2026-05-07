#!/usr/bin/env bash

export PYTHONPATH=/root/icefall:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$1

lr=0.001

# finetune checkpoint
do_finetune=1
finetune_ckpt=download/stage1-epoch-45-avg-28.pt

output_ds=2
post_output_ds=1

freeze_encoder=0
freeze_encoder_steps=-1
encoder_lr_scale=1

md=800

exp_dir=spear_roberta/exp_ft

echo $exp_dir

if false; then
python spear_roberta/finetune_stage2.py \
    --world-size 8 \
    --num-epochs 400 \
    --use-fp16 0 \
    --use-bf16 1 \
    --start-epoch 1 \
    --exp-dir $exp_dir \
    --manifest-dir data/manifests \
    --base-lr $lr \
    --do-finetune $do_finetune --finetune-ckpt $finetune_ckpt \
    --freeze-encoder $freeze_encoder --freeze-encoder-steps $freeze_encoder_steps \
    --encoder-lr-scale $encoder_lr_scale \
    --downsampling-factor 1,2,4,8,4,2,1 \
    --num-encoder-layers 1,2,3,4,1,1,1 \
    --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
    --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
    --encoder-unmasked-dim 768,768,768,768,768,768,768 \
    --cnn-module-kernel 31,31,15,15,15,31,31 \
    --num-heads 8,8,8,8,8,8,8 \
    --output-downsampling-factor $output_ds \
    --post-encoder-downsampling-factor $post_output_ds \
    --on-the-fly-feats 1 \
    --enable-musan 0 \
    --enable-spec-aug 0 \
    --max-duration $md
fi

if false; then
epoch=$2
# avg=$3
for epoch in $(seq $epoch 5 $((epoch + 24))); do
for avg in $(seq 2 5 $((epoch - 1))); do
  python spear_roberta/evaluate_retrieval.py \
      --epoch $epoch \
      --avg $avg \
      --manifest-dir data/manifests \
      --use-averaged-model 1 \
      --downsampling-factor 1,2,4,8,4,2,1 \
      --num-encoder-layers 1,2,3,4,1,1,1 \
      --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
      --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
      --encoder-unmasked-dim 768,768,768,768,768,768,768 \
      --cnn-module-kernel 31,31,15,15,15,31,31 \
      --num-heads 8,8,8,8,8,8,8 \
      --output-downsampling-factor $output_ds \
      --post-encoder-downsampling-factor $post_output_ds \
      --on-the-fly-feats 1 \
      --exp-dir $exp_dir \
      --max-duration $md
done
done
fi

if true; then
epoch=$2
avg=$3
# while read -r score tag; do
  # epoch=$(echo "$tag" | awk -F'[-]' '{print $2}')
  # avg=$(echo "$tag" | awk -F'[-]' '{print $4}')
  python spear_roberta/evaluate_zero_shot_classification.py \
      --epoch $epoch \
      --avg $avg \
      --manifest-dir data/manifests \
      --use-averaged-model 1 \
      --downsampling-factor 1,2,4,8,4,2,1 \
      --num-encoder-layers 1,2,3,4,1,1,1 \
      --feedforward-dim 3840,3840,3840,3840,3840,3840,3840 \
      --encoder-dim 1280,1280,1280,1280,1280,1280,1280 \
      --encoder-unmasked-dim 768,768,768,768,768,768,768 \
      --cnn-module-kernel 31,31,15,15,15,31,31 \
      --num-heads 8,8,8,8,8,8,8 \
      --output-downsampling-factor $output_ds \
      --post-encoder-downsampling-factor $post_output_ds \
      --on-the-fly-feats 1 \
      --exp-dir $exp_dir \
      --max-duration $md
# done < "$2"
fi

# for i in {0..7}; do CUDA_VISIBLE_DEVICES=$i python /root/busygpu/run.py & done
# python /root/busygpu/run.py &
