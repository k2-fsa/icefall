## Results

### WenetSpeech char-based training results (Pruned Transducer 2)

#### 2022-05-19

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/349.

When training with the L subset, the WERs are

|                                    |  dev  | test-net | test-meeting | comment                                  |
|------------------------------------|-------|----------|--------------|------------------------------------------|
|          greedy search             | 7.80  | 8.75     | 13.49        | --epoch 10, --avg 2, --max-duration 100  |
| modified beam search (beam size 4) | 7.76  | 8.71     | 13.41        | --epoch 10, --avg 2, --max-duration 100  |
| fast beam search (1best)  | 7.94  | 8.74     | 13.80        | --epoch 10, --avg 2, --max-duration 1500 |
| fast beam search (nbest)  | 9.82  | 10.98    |     16.37   | --epoch 10, --avg 2, --max-duration 600 |
| fast beam search (nbest oracle)  | 6.88 | 7.18    |     11.77   | --epoch 10, --avg 2, --max-duration 600 |
| fast beam search (nbest LG)  | 14.94 | 16.14    |   22.93  | --epoch 10, --avg 2, --max-duration 600 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless2/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless2/exp \
  --world-size 8 \
  --num-epochs 15 \
  --start-epoch 0 \
  --max-duration 180 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --training-subset L
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/wM4ZUNtASRavJx79EOYYcg/#scalars

The decoding command is:
```
epoch=10
avg=2

## greedy search
./pruned_transducer_stateless2/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method greedy_search

## modified beam search
./pruned_transducer_stateless2/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method modified_beam_search \
        --beam-size 4

## fast beam search
./pruned_transducer_stateless2/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8

## fast beam search (nbest)
./pruned_transducer_stateless2/decode.py \
        --epoch 10 \
        --avg 2 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 600 \
        --decoding-method fast_beam_search_nbest \
        --beam 20.0 \
        --max-contexts 8 \
        --max-states 64 \
        --num-paths 200 \
        --nbest-scale 0.5

## fast beam search (nbest oracle WER)
./pruned_transducer_stateless2/decode.py \
        --epoch 10 \
        --avg 2 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 600 \
        --decoding-method fast_beam_search_nbest_oracle \
        --beam 20.0 \
        --max-contexts 8 \
        --max-states 64 \
        --num-paths 200 \
        --nbest-scale 0.5

## fast beam search (with LG)
./pruned_transducer_stateless2/decode.py \
        --epoch 10 \
        --avg 2 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 600 \
        --decoding-method fast_beam_search_nbest_LG \
        --beam 20.0 \
        --max-contexts 8 \
        --max-states 64
```

When training with the M subset, the WERs are

|                                    |   dev  | test-net  | test-meeting  | comment                                   |
|------------------------------------|--------|-----------|---------------|-------------------------------------------|
|          greedy search             | 10.40  | 11.31     | 19.64         | --epoch 29, --avg 11, --max-duration 100  |
| modified beam search (beam size 4) |  9.85  | 11.04     | 18.20         | --epoch 29, --avg 11, --max-duration 100  |
| fast beam search (set as default)  | 10.18  | 11.10     | 19.32         | --epoch 29, --avg 11, --max-duration 1500 |


When training with the S subset, the WERs are

|                                    |  dev   | test-net  | test-meeting  | comment                                   |
|------------------------------------|--------|-----------|---------------|-------------------------------------------|
|          greedy search             | 19.92  | 25.20     | 35.35         | --epoch 29, --avg 24, --max-duration 100  |
| modified beam search (beam size 4) | 18.62  | 23.88     | 33.80         | --epoch 29, --avg 24, --max-duration 100  |
| fast beam search (set as default)  | 19.31  | 24.41     | 34.87         | --epoch 29, --avg 24, --max-duration 1500 |


A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2>
