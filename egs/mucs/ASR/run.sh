#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

./conformer_ctc/train.py \
    --num-epochs 60 \
    --max-duration 300 \
    --exp-dir ./conformer_ctc/exp_with_devset_split_bpe400 \
    --lang-dir data/lang_bpe_400 \
    --enable-musan False \


./conformer_ctc/decode.py \
    --epoch 59 \
    --avg 10 \
    --exp-dir ./conformer_ctc/exp_with_devset_split_bpe400 \
    --max-duration 100 \
    --lang-dir ./data/lang_bpe_400
