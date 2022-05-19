## Results

### Aidatatang_200zh Char training results (Pruned Transducer Stateless2)

#### 2022-05-16

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/375.

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 5.53       | 6.59       | --epoch 29, --avg 19, --max-duration 100 |
| modified beam search (beam size 4) | 5.27       | 6.33       | --epoch 29, --avg 19, --max-duration 100 |
| fast beam search (set as default)  | 5.30       | 6.34       | --epoch 29, --avg 19, --max-duration 1500|

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1"

./pruned_transducer_stateless2/train.py \
  --world-size 2 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless2/exp \
  --lang-dir data/lang_char \
  --max-duration 250 \
  --save-every-n 1000

```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/xS7kgYf2RwyDpQAOdS8rAA/#scalars

The decoding command is:
```
epoch=29
avg=19

## greedy search
./pruned_transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless2/exp \
  --lang-dir ./data/lang_char \
  --max-duration 100

## modified beam search
./pruned_transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless2/exp \
  --lang-dir ./data/lang_char \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4

## fast beam search
./pruned_transducer_stateless2/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir ./data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_aidatatang-200zh_pruned_transducer_stateless2>
