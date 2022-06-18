## Results

### Aishell4 Char training results (Pruned Transducer Stateless5)

#### 2022-06-13

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/399.

When use-averaged-model=False, the CERs are
|                                    |    test    | comment                                  |
|------------------------------------|------------|------------------------------------------|
|          greedy search             | 30.05      | --epoch 30, --avg 25, --max-duration 800 |
| modified beam search (beam size 4) | 29.16      | --epoch 30, --avg 25, --max-duration 800 |
| fast beam search (set as default)  | 29.20      | --epoch 30, --avg 25, --max-duration 1500|

When use-averaged-model=True, the CERs are
|                                    |    test    | comment                                                              |
|------------------------------------|------------|----------------------------------------------------------------------|
|          greedy search             | 29.89      | --iter 36000, --avg 8, --max-duration 800  --use-averaged-model=True |
| modified beam search (beam size 4) | 28.91      | --iter 36000, --avg 8, --max-duration 800  --use-averaged-model=True |
| fast beam search (set as default)  | 29.08      | --iter 36000, --avg 8, --max-duration 1500 --use-averaged-model=True |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir data/lang_char \
  --max-duration 220 \
  --save-every-n 4000

```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/tjaVRKERS8C10SzhpBcxSQ/#scalars

When use-averaged-model=False, the decoding command is:
```
epoch=30
avg=25

## greedy search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800

## modified beam search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800 \
  --decoding-method modified_beam_search \
  --beam-size 4

## fast beam search
./pruned_transducer_stateless5/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless5/exp \
        --lang-dir ./data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
```

When use-averaged-model=True, the decoding command is:
```
iter=36000
avg=8

## greedy search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800 \
  --use-averaged-model True

## modified beam search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800 \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --use-averaged-model True

## fast beam search
./pruned_transducer_stateless5/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless5/exp \
        --lang-dir ./data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8 \
        --use-averaged-model True
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_aishell4_pruned_transducer_stateless5>
