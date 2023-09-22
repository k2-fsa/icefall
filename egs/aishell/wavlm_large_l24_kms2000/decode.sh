for ((epoch=110; epoch<=120; epoch=$epoch+1)); do
  for ((avg=40; avg<=80; avg=$avg+1)); do
    ./zipformer/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir ./zipformer/exp \
      --max-duration 2000 \
      --context-size 1
  done
done
