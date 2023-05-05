#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

python ./pruned_transducer_stateless7/train.py \
    --world-size 1 \
    --start-epoch 1 \
    --exp-dir ./pruned_transducer_stateless7/exp-debug \
    --use-fp16 True \
    --num-epochs 40 \
    --subset medium \
    --manifest-dir data/fbank \
    --max-duration 500 \
    --master-port 13782
