## Results

### Alimeeting Char training results (Pruned Transducer Stateless2)

#### 2022-06-01

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/378.

The WERs are
|                                    |     eval   |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 31.77      | 34.66      | --epoch 29, --avg 18, --max-duration 100 |
| modified beam search (beam size 4) | 30.38      | 33.02      | --epoch 29, --avg 18, --max-duration 100 |
| fast beam search (set as default)  | 31.39      | 34.25      | --epoch 29, --avg 18, --max-duration 1500|

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless2/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless2/exp \
  --lang-dir data/lang_char \
  --max-duration 220 \
  --save-every-n 1000

```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/AoqgSvZKTZCJhJbOuG3W6g/#scalars

The decoding command is:
```
epoch=29
avg=18

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

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_alimeeting_pruned_transducer_stateless2>
