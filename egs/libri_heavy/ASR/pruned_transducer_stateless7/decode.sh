#!/usr/bin/env bash

. /ceph-data4/yangxiaoyu/softwares/anaconda3/etc/profile.d/conda.sh && conda deactivate && conda activate k2_latest

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=~/softwares/icefall_development/icefall_libri_light:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="0"

echo "Using device: ${CUDA_VISIBLE_DEVICES}"

for avg in 10; do
    ./pruned_transducer_stateless7/decode.py \
        --epoch 60 \
        --avg $avg \
        --use-averaged-model True \
        --exp-dir ./pruned_transducer_stateless7/exp_small_uppercase_ref_4gpus \
        --bpe-model /ceph-data4/yangxiaoyu/softwares/icefall_development/icefall_libri_light/egs/libri_heavy/ASR/data/lang_bpe_500/bpe.model \
        --max-duration 1200 \
        --decoding-method greedy_search
done
