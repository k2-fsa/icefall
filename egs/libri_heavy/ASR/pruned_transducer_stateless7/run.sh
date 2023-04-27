#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2,3,4,5"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

python ./pruned_transducer_stateless7/train.py \
    --world-size 4 \
    --start-epoch 31 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --use-fp16 True \
    --num-epochs 60 \
    --subset small \
    --manifest-dir data/fbank \
    --max-duration 500 \
    --master-port 13789
