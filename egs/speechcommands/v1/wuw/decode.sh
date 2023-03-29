export CUDA_VISIBLE_DEVICES=5

for epoch in {21..30}; do
  for ((avg=1; avg<=5; avg++)); do
    ./tdnn/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./tdnn/exp \
      --max-duration 2000 \
      --decoding-method greedy_search
  done
done
