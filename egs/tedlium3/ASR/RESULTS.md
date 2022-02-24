## Results

### TedLium3 BPE training results (Transducer)

#### Conformer encoder + embedding decoder

Using the codes from this PR commit https://github.com/k2-fsa/icefall/pull/183/commits/536ad2252e2d406f24a681743d98bd5f90801b97.

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 7.31       | 6.73       | --epoch 71, --avg 15, --max-duration 100 |
|      beam search (beam size 4)     | 7.12       | 6.58       | --epoch 71, --avg 15, --max-duration 100 |
| modified beam search (beam size 4) | 7.20       | 6.65       | --epoch 71, --avg 15, --max-duration 100 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp \
  --max-duration 180 \
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/DnRwoZF8RRyod4kkfG5q5Q/#scalars

The decoding command is:
```
epoch=29
avg=15

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
