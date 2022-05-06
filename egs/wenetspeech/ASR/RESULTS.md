## Results

### WenetSpeech char-based training results (Pruned Transducer 2)

#### 2022-05-06

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/349 and the Lhotse v1.1.

When training with the L subset, the WERs are

|                                    |  dev  | test-net | test-meeting | comment                                 |
|------------------------------------|-------|----------|--------------|-----------------------------------------|
|          greedy search             | 8.06  | 9.16     | 14.07        | --epoch 6, --avg 3, --max-duration 100  |
| modified beam search (beam size 4) | 7.97  | 9.18     | 13.91        | --epoch 6, --avg 3, --max-duration 100  |
| fast beam search (set as default)  | 8.13  | 9.12     | 14.33        | --epoch 6, --avg 3, --max-duration 1500 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless2/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless2/exp \
  --world-size 8 \
  --num-epochs 10 \
  --start-epoch 0 \
  --max-duration 180 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --training-subset L
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/VpA8b7SZQ7CEjZs9WZ5HNA/#scalars

The decoding command is:
```
epoch=6
avg=3

## greedy search
./pruned_transducer_stateless2/decode.py \
        --epoch 6 \
        --avg 3 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method greedy_search

## modified beam search
./pruned_transducer_stateless2/decode.py \
        --epoch 6 \
        --avg 3 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 100 \
        --decoding-method modified_beam_search \
        --beam-size 4

## fast beam search
./pruned_transducer_stateless2/decode.py \
        --epoch 6 \
        --avg 3 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --lang-dir data/lang_char \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
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
