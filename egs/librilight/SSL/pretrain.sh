export PYTHONPATH=$(pwd)/../../..

./zipformer/pretrain.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp_pretrain \
  --max-duration 650 \
  --quadratic-duration 512 \
  --accum-grad 1 \
  --do-normalize 1 \
  --mask-prob 0.8 \
  --extractor-mode "layer_norm" \
  --dropout-input 0.0 \
  --dropout-features 0.0 \
  --feature-grad-mult 1.0 \
  --num-encoder-layers 2,2,3,4,3,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,448,768,448,192 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --base-lr 0.045
