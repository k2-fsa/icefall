## Results

### WenetSpeech char-based training results (Non-streaming and streaming) on zipformer model

This is the [pull request](https://github.com/k2-fsa/icefall/pull/1130) in icefall.

#### Non-streaming

Best results (num of params : ~76M):

Type | Greedy(dev & net & meeting) | Beam search(dev & net & meeting) |  
-- | -- | -- | --
Non-streaming | 7.36 & 7.65 & 12.43 | 7.32 & 7.61 & 12.35 | --epoch=12

The training command:

```
./zipformer/train.py \
  --world-size 6 \
  --num-epochs 12 \
  --use-fp16 1 \
  --max-duration 450 \
  --training-subset L \
  --lr-epochs 1.5 \
  --context-size 2 \
  --exp-dir zipformer/exp_L_context_2 \
  --causal 0 \
  --num-workers 8
```

Listed best results for each epoch below:

Epoch | Greedy search(dev & net & meeting) | Modified beam search(dev & net & meeting) |  
-- | -- | -- | --
4 | 7.83 & 8.86 &13.73 | 7.75 & 8.81 & 13.67 | avg=1;blank-penalty=2
5 | 7.75 & 8.46 & 13.38 | 7.68 & 8.41 & 13.27 | avg=1;blank-penalty=2
6 | 7.72 & 8.19 & 13.16 | 7.62 & 8.14 & 13.06 | avg=1;blank-penalty=2
7 | 7.59 & 8.08 & 12.97 | 7.53 & 8.01 & 12.87 | avg=2;blank-penalty=2
8 | 7.68 & 7.87 & 12.96 | 7.61 & 7.81 & 12.88 | avg=1;blank-penalty=2
9 | 7.57 & 7.77 & 12.87 | 7.5 & 7.71 & 12.77 | avg=1;blank-penalty=2
10 | 7.45 & 7.7 & 12.69 | 7.39 & 7.63 & 12.59 | avg=2;blank-penalty=2
11 | 7.35 & 7.67 & 12.46 | 7.31 & 7.63 & 12.43 | avg=3;blank-penalty=2
12 | 7.36 & 7.65 & 12.43 | 7.32 & 7.61 & 12.35 | avg=4;blank-penalty=2

The pre-trained model is available here : https://huggingface.co/pkufool/icefall-asr-zipformer-wenetspeech-20230615


#### Streaming

Best results (num of params : ~76M):

Type | Greedy(dev & net & meeting) | Beam search(dev & net & meeting) |  
-- | -- | -- | --
Streaming | 8.45 & 9.89 & 16.46 | 8.21 & 9.77 & 16.07 | --epoch=12; --chunk-size=16; --left-context-frames=256
Streaming | 8.0 & 9.0 & 15.11 | 7.84 & 8.94 & 14.92 | --epoch=12; --chunk-size=32; --left-context-frames=256

The training command:

```
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 12 \
  --use-fp16 1 \
  --max-duration 450 \
  --training-subset L \
  --lr-epochs 1.5 \
  --context-size 2 \
  --exp-dir zipformer/exp_L_causal_context_2 \
  --causal 1 \
  --num-workers 8
```

Best results for each epoch (--chunk-size=16; --left-context-frames=128)

Epoch | Greedy search(dev & net & meeting) | Modified beam search(dev & net & meeting) |  
-- | -- | -- | --
6 | 9.14 & 10.75 & 18.15 | 8.79 & 10.54 & 17.64 | avg=1;blank-penalty=1.5
7 | 9.11 & 10.61 & 17.86 | 8.8 & 10.42 & 17.29 | avg=1;blank-penalty=1.5
8 | 8.89 & 10.32 & 17.44 | 8.59 & 10.09 & 16.9 | avg=1;blank-penalty=1.5
9 | 8.86 & 10.11 & 17.35 | 8.55 & 9.87 & 16.76 | avg=1;blank-penalty=1.5
10 | 8.66 & 10.0 & 16.94 | 8.39 & 9.83 & 16.47 | avg=2;blank-penalty=1.5
11 | 8.58 & 9.92 & 16.67 | 8.32 & 9.77 & 16.27 | avg=3;blank-penalty=1.5
12 | 8.45 & 9.89 & 16.46 | 8.21 & 9.77 & 16.07 | avg=4;blank-penalty=1.5

The pre-trained model is available here: https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615


### WenetSpeech char-based training results (offline and streaming) (Pruned Transducer 5)

#### 2022-07-22

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/447.

When training with the L subset, the CERs are

**Offline**:
|decoding-method| epoch | avg | use-averaged-model | DEV | TEST-NET | TEST-MEETING|
|-- | -- | -- | -- | -- | -- | --|
|greedy_search | 4 | 1 | True | 8.22 | 9.03 | 14.54|
|modified_beam_search | 4 | 1 | True | **8.17** | **9.04** | **14.44**|
|fast_beam_search | 4 | 1 | True | 8.29 | 9.00 | 14.93|

The offline training command for reproducing is given below:
```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless5/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless5/exp_L_offline \
  --world-size 8 \
  --num-epochs 15 \
  --start-epoch 2 \
  --max-duration 120 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --average-period 1000 \
  --training-subset L
```

The tensorboard training log can be found at https://tensorboard.dev/experiment/SvnN2jfyTB2Hjqu22Z7ZoQ/#scalars .


A pre-trained offline model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_offline>

**Streaming**:
|decoding-method| epoch | avg | use-averaged-model | DEV | TEST-NET | TEST-MEETING|
|--|--|--|--|--|--|--|
| greedy_search | 7| 1| True | 8.78 | 10.12 | 16.16 |
| modified_beam_search | 7| 1| True| **8.53**| **9.95** | **15.81** |
| fast_beam_search | 7 | 1| True | 9.01 | 10.47 | 16.28 |

The streaming training command for reproducing is given below:
```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless5/train.py \
  --lang-dir data/lang_char \
  --exp-dir pruned_transducer_stateless5/exp_L_streaming \
  --world-size 8 \
  --num-epochs 15 \
  --start-epoch 1 \
  --max-duration 140 \
  --valid-interval 3000 \
  --model-warm-step 3000 \
  --save-every-n 8000 \
  --average-period 1000 \
  --training-subset L \
  --dynamic-chunk-training True \
  --causal-convolution True \
  --short-chunk-size 25 \
  --num-left-chunks 4
```

The tensorboard training log can be found at https://tensorboard.dev/experiment/E2NXPVflSOKWepzJ1a1uDQ/#scalars .


A pre-trained offline model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless5_streaming>

### WenetSpeech char-based training results (Pruned Transducer 2)

#### 2022-05-19

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/349.

When training with the L subset, the CERs are

|                                    |  dev  | test-net | test-meeting | comment                                  |
|------------------------------------|-------|----------|--------------|------------------------------------------|
|          greedy search             | 7.80  | 8.75     | 13.49        | --epoch 10, --avg 2, --max-duration 100  |
| modified beam search (beam size 4) | 7.76  | 8.71     | 13.41        | --epoch 10, --avg 2, --max-duration 100  |
| fast beam search (1best)  | 7.94  | 8.74     | 13.80        | --epoch 10, --avg 2, --max-duration 1500 |
| fast beam search (nbest)  | 9.82  | 10.98    |     16.37   | --epoch 10, --avg 2, --max-duration 600 |
| fast beam search (nbest oracle)  | 6.88 | 7.18    |     11.77   | --epoch 10, --avg 2, --max-duration 600 |
| fast beam search (nbest LG, ngram_lm_scale=0.35)  | 8.83 | 9.88    |   15.47  | --epoch 10, --avg 2, --max-duration 600 |

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

## fast beam search (1best)
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
        --ngram-lm-scale 0.35 \
        --beam 20.0 \
        --max-contexts 8 \
        --max-states 64
```

When training with the M subset, the CERs are

|                                    |   dev  | test-net  | test-meeting  | comment                                   |
|------------------------------------|--------|-----------|---------------|-------------------------------------------|
|          greedy search             | 10.40  | 11.31     | 19.64         | --epoch 29, --avg 11, --max-duration 100  |
| modified beam search (beam size 4) |  9.85  | 11.04     | 18.20         | --epoch 29, --avg 11, --max-duration 100  |
| fast beam search (set as default)  | 10.18  | 11.10     | 19.32         | --epoch 29, --avg 11, --max-duration 1500 |


When training with the S subset, the CERs are

|                                    |  dev   | test-net  | test-meeting  | comment                                   |
|------------------------------------|--------|-----------|---------------|-------------------------------------------|
|          greedy search             | 19.92  | 25.20     | 35.35         | --epoch 29, --avg 24, --max-duration 100  |
| modified beam search (beam size 4) | 18.62  | 23.88     | 33.80         | --epoch 29, --avg 24, --max-duration 100  |
| fast beam search (set as default)  | 19.31  | 24.41     | 34.87         | --epoch 29, --avg 24, --max-duration 1500 |


A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_wenetspeech_pruned_transducer_stateless2>
