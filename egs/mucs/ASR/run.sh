#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

./conformer_ctc/train.py \
    --num-epochs 60 \
    --max-duration 100 \
    --exp-dir ./conformer_ctc/exp \
    --lang-dir data/lang_bpe_200 \
    --enable-musan False \

# ./conformer_ctc/decode.py \
#     --epoch 59 \
#     --avg 10 \
#     --exp-dir ./conformer_ctc/exp \
#     --max-duration 100 \
#     --lang-dir ./data/lang_bpe_2000
