#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

set -e 
dataset='hi-en'
datadir=data_"$dataset"
bpe=400
decode_methods="attention-decoder 1best nbest ctc-decoding whole-lattice-rescoring"
num_paths=20
max_duration=5

./conformer_ctc/train.py \
    --num-epochs 60 \
    --max-duration 300 \
    --exp-dir ./conformer_ctc/exp_"$dataset"_bpe"$bpe" \
    --manifest-dir $datadir/fbank \
    --lang-dir $datadir/lang_bpe_"$bpe" \
    --enable-musan False \

for decode_method in $decode_methods; 
do

    ./conformer_ctc/decode.py \
        --epoch 59 \
        --avg 10 \
        --manifest-dir $datadir/fbank \
        --exp-dir ./conformer_ctc/exp_"$dataset"_bpe"$bpe" \
        --max-duration $max_duration \
        --lang-dir $datadir/lang_bpe_"$bpe" \
        --lm-dir $datadir/"lm" \
        --method $decode_method \
        --num-paths $num_paths \
        
done
exit

    