#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

./pruned_transducer_stateless7/decode.py \
    --epoch 60 \
    --avg 10 \
    --use-averaged-model True \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method greedy_search
