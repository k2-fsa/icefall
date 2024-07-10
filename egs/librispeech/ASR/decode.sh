export CUDA_VISIBLE_DEVICES=2

for epoch in {30..30}; do
  for ((avg=1; avg<=$epoch-1; avg++)); do
    ./zipformer_lstm/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./zipformer_lstm/exp_dropout0.2 \
      --max-duration 2000 \
      --decoding-method greedy_search
  done
done
