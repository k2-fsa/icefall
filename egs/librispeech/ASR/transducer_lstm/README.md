## Introduction

The encoder consists of LSTM layers in this folder. You can use the
following command to start the training:

```bash
cd egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2"

./transducer_lstm/train.py \
  --world-size 3 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_lstm/exp \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 3
```
