#prepare dataset
#bash prepare_20230113.sh --stage 1 --stop_stage 1 >myout 2>&1 &

export PYTHONPATH=../../../../icefall
export CUDA_VISIBLE_DEVICES="4,5,6,7"
stage=1
stop_stage=1
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
./conv_emformer_transducer_stateless2/train.py \
  --world-size 4 \
  --num-epochs 100 \
  --lang-dir data/lang_char \
  --start-epoch 38 \
  --use-fp16 0 \
  --exp-dir exp_conv_emformer \
  --max-duration 30 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32
fi

# export-for-ncnn
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ./conv_emformer_transducer_stateless2/export-for-ncnn.py \
  --exp-dir exp_conv_emformer \
  --lang_dir data/lang_char \
  --epoch 39 \
  --avg 10 \
  --use-averaged-model 1 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32
fi

# export pre-model
epoch=30
avg=10
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
./conv_emformer_transducer_stateless2/export.py \
  --exp-dir exp_conv_emformer \
  --epoch $epoch \
  --avg $avg \
  --use-averaged-model=True \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --jit 0
fi
