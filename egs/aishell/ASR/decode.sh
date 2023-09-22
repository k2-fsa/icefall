./zipformer/decode.py \
  --epoch 40 \
  --avg 12 \
  --exp-dir ./zipformer/exp \
  --max-duration 2000 \
  --decoding-method modified_beam_search \
  --beam-size 4
