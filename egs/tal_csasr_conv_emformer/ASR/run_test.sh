export PYTHONPATH=../../../../icefall
export CUDA_VISIBLE_DEVICES="4"
./conv_emformer_transducer_stateless2/decode_db.py \
      --epoch 2 \
      --avg 1 \
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
