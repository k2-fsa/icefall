## Results

### TedLium3 BPE training results (Zipformer)

#### 2023-06-15 (Regular transducer)

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/1125.

Number of model parameters: 65549011, i.e., 65.5 M

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 6.74       | 6.16       | --epoch 50, --avg 22, --max-duration 500 |
|      beam search (beam size 4)     | 6.56       | 5.95       | --epoch 50, --avg 22, --max-duration 500 |
| modified beam search (beam size 4) | 6.54       | 6.00       | --epoch 50, --avg 22, --max-duration 500 |
| fast beam search (set as default)  | 6.91       | 6.28       | --epoch 50, --avg 22, --max-duration 500 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer/train.py \
  --use-fp16 true \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 0 \
  --exp-dir zipformer/exp \
  --max-duration 1000
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/AKXbJha0S9aXyfmuvG4h5A/#scalars

The decoding command is:
```
epoch=50
avg=22

## greedy search
./zipformer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir zipformer/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 500

## beam search
./zipformer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir zipformer/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 500 \
  --decoding-method beam_search \
  --beam-size 4

## modified beam search
./zipformer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir zipformer/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 500 \
  --decoding-method modified_beam_search \
  --beam-size 4

## fast beam search
./zipformer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./zipformer/exp \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 1500 \
  --decoding-method fast_beam_search \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/desh2608/icefall-asr-tedlium3-zipformer>

#### 2023-06-26 (Modified transducer)

```
./zipformer/train.py \
  --use-fp16 true \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 0 \
  --exp-dir zipformer/exp \
  --max-duration 1000 \
  --rnnt-type modified
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/3d4bYmbJTGiWQQaW88CVEQ/#scalars

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 6.32       | 5.83       | --epoch 50, --avg 22, --max-duration 500 |
| modified beam search (beam size 4) | 6.16       | 5.79       | --epoch 50, --avg 22, --max-duration 500 |
| fast beam search (set as default)  | 6.30       | 5.89       | --epoch 50, --avg 22, --max-duration 500 |

A pre-trained model and decoding logs can be found at <https://huggingface.co/desh2608/icefall-asr-tedlium3-zipformer>.

### TedLium3 BPE training results (Conformer-CTC 2)

#### [conformer_ctc2](./conformer_ctc2)

See <https://github.com/k2-fsa/icefall/pull/696> for more details.

The tensorboard log can be found at
<https://tensorboard.dev/experiment/5NQQiqOqSqazfn4w2yeWEQ/>

You can find a pretrained model and decoding results at:
<https://huggingface.co/videodanchik/icefall-asr-tedlium3-conformer-ctc2>

Number of model parameters: 101141699, i.e., 101.14 M

The WERs are

|                          | dev        | test        | comment             |
|--------------------------|------------|-------------|---------------------|
| ctc decoding             | 6.45       | 5.96        | --epoch 38 --avg 26 |
| 1best                    | 5.92       | 5.51        | --epoch 38 --avg 26 |
| whole lattice rescoring  | 5.96       | 5.47        | --epoch 38 --avg 26 |
| attention decoder        | 5.60       | 5.33        | --epoch 38 --avg 26 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./conformer_ctc2/train.py \
    --world-size 4 \
    --num-epochs 40 \
    --exp-dir conformer_ctc2/exp \
    --max-duration 350 \
    --use-fp16 true
```

The decoding command is:
```
epoch=38
avg=26

## ctc decoding
./conformer_ctc2/decode.py \
  --method ctc-decoding \
  --exp-dir conformer_ctc2/exp \
  --lang-dir data/lang_bpe_500 \
  --result-dir conformer_ctc2/exp \
  --max-duration 500 \
  --epoch $epoch \
  --avg $avg

## 1best
./conformer_ctc2/decode.py \
  --method 1best \
  --exp-dir conformer_ctc2/exp \
  --lang-dir data/lang_bpe_500 \
  --result-dir conformer_ctc2/exp \
  --max-duration 500 \
  --epoch $epoch \
  --avg $avg

## whole lattice rescoring
./conformer_ctc2/decode.py \
  --method whole-lattice-rescoring \
  --exp-dir conformer_ctc2/exp \
  --lm-path data/lm/G_4_gram_big.pt \
  --lang-dir data/lang_bpe_500 \
  --result-dir conformer_ctc2/exp \
  --max-duration 500 \
  --epoch $epoch \
  --avg $avg

## attention decoder
./conformer_ctc2/decode.py \
  --method attention-decoder \
  --exp-dir conformer_ctc2/exp \
  --lang-dir data/lang_bpe_500 \
  --result-dir conformer_ctc2/exp \
  --max-duration 500 \
  --epoch $epoch \
  --avg $avg
```

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
