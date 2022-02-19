## Introduction

The decoder, i.e., the prediction network, is from
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
(Rnn-Transducer with Stateless Prediction Network)

You can use the following command to start the training:

```bash
cd egs/aishell/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./transducer_stateless_modified/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_modified/exp \
  --max-duration 250 \
  --lr-factor 2.5
```
