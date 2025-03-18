## Introduction

The decoder, i.e., the prediction network, is from
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
(Rnn-Transducer with Stateless Prediction Network)

The auxiliary losses are from https://arxiv.org/pdf/2011.03109v2.pdf
(Improving RNN transducer based ASR with auxiliary tasks)


You can use the following command to start the training:

```bash
cd egs/librispeech/ASR
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1"

./transducer_stateless_aux_kl/train.py \
  --world-size 2 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_aux_kl/exp-100 \
  --full-libri 0 \
  --max-duration 300 \
  --lr-factor 1 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25
  --giga-prob 0.2
```
