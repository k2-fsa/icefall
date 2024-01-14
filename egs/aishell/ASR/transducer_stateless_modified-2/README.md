## Introduction

The decoder, i.e., the prediction network, is from
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
(Rnn-Transducer with Stateless Prediction Network)

Different from `../transducer_stateless_modified`, this folder
uses extra data, i.e., http://www.openslr.org/62/, during training.

You can use the following command to start the training:

```bash
cd egs/aishell/ASR
./prepare.sh --stop-stage 6
./prepare_aidatatang_200zh.sh

export CUDA_VISIBLE_DEVICES="0,1,2"

./transducer_stateless_modified-2/train.py \
  --world-size 3 \
  --num-epochs 90 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_modified-2/exp-2 \
  --max-duration 250 \
  --lr-factor 2.0 \
  --context-size 2 \
  --modified-transducer-prob 0.25 \
  --datatang-prob 0.2
```

To decode, you can use

```bash
for epoch in 89; do
  for avg in 30 38; do
  ./transducer_stateless_modified-2/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified-2/exp-2 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method greedy_search \
    --max-sym-per-frame 1
  done
done

for epoch in 89; do
  for avg in 38; do
    ./transducer_stateless_modified-2/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified-2/exp-2 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method modified_beam_search \
    --beam-size 4
  done
done
```
