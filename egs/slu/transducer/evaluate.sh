#!/usr/bin/env bash

# exp_dir=/home/xli257/slu/icefall_st/egs/slu/transducer/exp_norm_30_01_50_5/rank_reverse/percentage2_snr30
exp_dir=$1

# feature_dir=/home/xli257/slu/icefall_st/egs/slu/data/icefall_non_adv_0/percentage1_snr20/fbanks
feature_dir=$2

epoch=6

conda activate slu_icefall

cd /home/xli257/slu/icefall_st/egs/slu/

CUDA_VISIBLE_DEVICES=$(free-gpu) python /home/xli257/slu/icefall_st/egs/slu/transducer/decode.py --epoch $epoch --exp-dir $exp_dir  --feature-dir $feature_dir
