#!/usr/bin/env bash

# exp_dir=transducer/exp_norm_30_01_50_5/rank_reverse/instance40_snr20
exp_dir=$1

# feature_dir=data/norm_30_01_50_5/rank_reverse/instance40_snr20/fbanks
feature_dir=$2

seed=0

conda activate slu_icefall

cd /home/xli257/slu/icefall_st/egs/slu/

CUDA_VISIBLE_DEVICES=$(free-gpu) ./transducer/train.py --exp-dir $exp_dir --lang-dir data/icefall_adv/percentage5_scale01/lm/frames --seed $seed --feature-dir $feature_dir
