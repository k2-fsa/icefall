#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

subset=small
python ./pruned_transducer_stateless7/train.py \
    --world-size 4 \
    --start-epoch 7 \
    --exp-dir ./pruned_transducer_stateless7/exp-${subset}-fixed-start-full-width-md450-4gpus \
    --use-fp16 True \
    --num-epochs 60 \
    --subset $subset \
    --manifest-dir data/fbank_new \
    --max-duration 450 \
    --master-port 13780
