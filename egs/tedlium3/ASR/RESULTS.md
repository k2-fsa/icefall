## Results

### TedLium3 BPE training results (Transducer)

#### Conformer encoder + embedding decoder

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/233
And the SpecAugment codes from this PR https://github.com/lhotse-speech/lhotse/pull/604

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 7.19       | 6.57       | --epoch 29, --avg 16, --max-duration 100 |
|      beam search (beam size 4)     | 7.12       | 6.37       | --epoch 29, --avg 16, --max-duration 100 |
| modified beam search (beam size 4) | 7.00       | 6.19       | --epoch 29, --avg 16, --max-duration 100 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp \
  --max-duration 200 \
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/zrfXeJO3Q5GmJpP2KRd2VA/#scalars

The decoding command is:
```
epoch=29
avg=16

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
  --decoding-method beam_search \
  --beam-size 4
```
