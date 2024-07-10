export CUDA_VISIBLE_DEVICES=$1

./zipformer_lstm/decode.py \
  --epoch $2 \
  --avg $3 \
  --exp-dir ./zipformer_lstm/exp \
  --max-duration 2000 \
  --decoding-method beam_search
