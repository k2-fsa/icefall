## Introduction

The encoder consists of Conformer layers in this folder. You can use the
following command to start the training:

```bash
cd egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer/exp \
  --full-libri 1 \
  --max-duration 250 \
  --lr-factor 2.5
```
