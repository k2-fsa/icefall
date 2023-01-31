## Results

### TedLium3 BPE training results (Pruned Transducer)

#### 2022-03-21

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/261.

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 7.27       | 6.69       | --epoch 29, --avg 13, --max-duration 100 |
|      beam search (beam size 4)     | 6.70       | 6.04       | --epoch 29, --avg 13, --max-duration 100 |
| modified beam search (beam size 4) | 6.77       | 6.14       | --epoch 29, --avg 13, --max-duration 100 |
| fast beam search (set as default)  | 7.14       | 6.50       | --epoch 29, --avg 13, --max-duration 1500|

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --max-duration 300
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/VpA8b7SZQ7CEjZs9WZ5HNA/#scalars

The decoding command is:
```
epoch=29
avg=13

## greedy search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100

## beam search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --decoding-method beam_search \
  --beam-size 4

## modified beam search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4

## fast beam search
./pruned_transducer_stateless/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless/exp \
        --bpe-model ./data/lang_bpe_500/bpe.model \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_tedlium3_pruned_transducer_stateless>

### TedLium3 BPE training results (Transducer)

#### Conformer encoder + embedding decoder

##### 2022-03-21

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/233
And the SpecAugment codes from this PR https://github.com/lhotse-speech/lhotse/pull/604

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 7.19       | 6.70       | --epoch 29, --avg 11, --max-duration 100 |
|      beam search (beam size 4)     | 7.02       | 6.36       | --epoch 29, --avg 11, --max-duration 100 |
| modified beam search (beam size 4) | 6.91       | 6.33       | --epoch 29, --avg 11, --max-duration 100 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp \
  --max-duration 300
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/4ks15jYHR4uMyvpW7Nz76Q/#scalars

The decoding command is:
```
epoch=29
avg=11

## greedy search
./transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100

## beam search
./transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --decoding-method beam_search \
  --beam-size 4

## modified beam search
./transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_tedlium3_transducer_stateless>
