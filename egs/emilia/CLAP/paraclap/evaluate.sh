#!/usr/bin/env bash

export PYTHONPATH=/root/icefall:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

md=800

exp_dir=paraclap/exp

echo $exp_dir

if false; then
python paraclap/evaluate_retrieval.py \
    --manifest-dir data/manifests \
    --on-the-fly-feats 1 \
    --exp-dir $exp_dir \
    --max-duration $md
fi

if true; then
python paraclap/evaluate_zero_shot_classification.py \
    --manifest-dir data/manifests \
    --on-the-fly-feats 1 \
    --exp-dir $exp_dir \
    --max-duration $md
fi

# python /root/busygpu/run.py &
