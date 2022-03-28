## Introduction

The decoder, i.e., the prediction network, is from
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
(Rnn-Transducer with Stateless Prediction Network)

You can use the following command to start the training:

```bash
cd egs/wenetspeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --token-type lazy_pinyin \
  --lang-dir data/lang_lazy_pinyin \
  --max-duration 250
```
