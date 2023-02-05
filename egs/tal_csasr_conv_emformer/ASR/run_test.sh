export PYTHONPATH=../../../../icefall
export CUDA_VISIBLE_DEVICES="4"

stage=2
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
./conv_emformer_transducer_stateless2/decode.py \
      --epoch 35 \
      --avg 5 \
      --use-averaged-model True \
      --exp-dir exp_conv_emformer \
      --max-duration 200 \
      --num-encoder-layers 12 \
      --chunk-length 32 \
      --cnn-module-kernel 31 \
      --left-context-length 32 \
      --right-context-length 8 \
      --memory-size 32 \
      --decoding-method fast_beam_search \
      --beam 4 \
      --max-contexts 4 \
      --max-states 8
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
./conv_emformer_transducer_stateless2/decode_db.py \
        --exp-dir exp_conv_emformer \
        --epoch 999 \
        --avg 1 \
        --max-duration 100 \
        --use-averaged-model=False \
        --num-encoder-layers 12 \
        --chunk-length 32 \
        --cnn-module-kernel 31 \
        --left-context-length 32 \
        --right-context-length 8 \
        --memory-size 32
fi
