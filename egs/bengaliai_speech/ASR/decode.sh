for ((avg=17; avg<=21; avg=$avg+1)); do
./zipformer/decode.py \
    --epoch 120 \
    --avg $avg \
    --exp-dir ./zipformer/exp \
    --max-duration 2000 \
    --decoding-method greedy_search
done
