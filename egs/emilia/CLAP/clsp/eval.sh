#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

md=800

python clsp/eval_zero_shot_classification.py \
      --manifest-dir data/manifests \
      --on-the-fly-feats 1 \
      --max-duration $md
