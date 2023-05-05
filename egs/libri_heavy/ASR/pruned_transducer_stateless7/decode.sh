#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

for avg in 10; do
    ./pruned_transducer_stateless7/decode.py \
        --epoch 40 \
        --avg $avg \
        --use-averaged-model True \
        --exp-dir ./pruned_transducer_stateless7/exp-medium-replace-full-width-4gpus \
        --max-duration 1200 \
        --random-left-padding False \
        --decoding-method greedy_search
done