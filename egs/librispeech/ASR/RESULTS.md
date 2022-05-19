## Results

### LibriSpeech BPE training results (Pruned Transducer 3, 2022-04-29)

[pruned_transducer_stateless3](./pruned_transducer_stateless3)
Same as `Pruned Transducer 2` but using the XL subset from
[GigaSpeech](https://github.com/SpeechColab/GigaSpeech) as extra training data.

During training, it selects either a batch from GigaSpeech with prob `giga_prob`
or a batch from LibriSpeech with prob `1 - giga_prob`. All utterances within
a batch come from the same dataset.

Using commit `ac84220de91dee10c00e8f4223287f937b1930b6`.

See <https://github.com/k2-fsa/icefall/pull/312>.

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|----------------------------------------|
| greedy search (max sym per frame 1) | 2.21       | 5.09       | --epoch 27 --avg 2  --max-duration 600 |
| greedy search (max sym per frame 1) | 2.25       | 5.02       | --epoch 27 --avg 12 --max-duration 600 |
| modified beam search                | 2.19       | 5.03       | --epoch 25 --avg 6  --max-duration 600 |
| modified beam search                | 2.23       | 4.94       | --epoch 27 --avg 10 --max-duration 600 |
| beam search                         | 2.16       | 4.95       | --epoch 25 --avg 7  --max-duration 600 |
| fast beam search                    | 2.21       | 4.96       | --epoch 27 --avg 10 --max-duration 600 |
| fast beam search                    | 2.19       | 4.97       | --epoch 27 --avg 12 --max-duration 600 |

The training commands are:

```bash
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless3/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 0 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless3/exp \
  --max-duration 300 \
  --use-fp16 1 \
  --lr-epochs 4 \
  --num-workers 2 \
  --giga-prob 0.8
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/gaD34WeYSMCOkzoo3dZXGg/>
(Note: The training process is killed manually after saving `epoch-28.pt`.)

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29>

The decoding commands are:

```bash

# greedy search
./pruned_transducer_stateless3/decode.py \
    --epoch 27 \
    --avg 2 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method greedy_search \
    --max-sym-per-frame 1

# modified beam search
./pruned_transducer_stateless3/decode.py \
    --epoch 25 \
    --avg 6 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --max-sym-per-frame 1

# beam search
./pruned_transducer_stateless3/decode.py \
    --epoch 25 \
    --avg 7 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --max-sym-per-frame 1

# fast beam search
for epoch in 27; do
  for avg in 10 12; do
    ./pruned_transducer_stateless3/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp \
        --max-duration 600 \
        --decoding-method fast_beam_search \
        --max-states 32 \
        --beam 8
  done
done
```

The following table shows the
[Nbest oracle WER](http://kaldi-asr.org/doc/lattices.html#lattices_operations_oracle)
for fast beam search.
| epoch | avg | num_paths | nbest_scale | test-clean | test-other |
|-------|-----|-----------|-------------|------------|------------|
|  27   | 10  |   50      | 0.5         |  0.91      |  2.74      |
|  27   | 10  |   50      | 0.8         |  0.94      |  2.82      |
|  27   | 10  |   50      | 1.0         |  1.06      |  2.88      |
|  27   | 10  |   100     | 0.5         |  0.82      |  2.58      |
|  27   | 10  |   100     | 0.8         |  0.92      |  2.65      |
|  27   | 10  |   100     | 1.0         |  0.95      |  2.77      |
|  27   | 10  |   200     | 0.5         |  0.81      |  2.50      |
|  27   | 10  |   200     | 0.8         |  0.85      |  2.56      |
|  27   | 10  |   200     | 1.0         |  0.91      |  2.64      |
|  27   | 10  |   400     | 0.5         |  N/A       |  N/A       |
|  27   | 10  |   400     | 0.8         |  0.81      |  2.49      |
|  27   | 10  |   400     | 1.0         |  0.85      |  2.54      |

The Nbest oracle WER is computed using the following steps:

  - 1. Use `fast_beam_search` to produce a lattice.
  - 2. Extract `N` paths from the lattice using [k2.random_path](https://k2-fsa.github.io/k2/python_api/api.html#random-paths)
  - 3. [Unique](https://k2-fsa.github.io/k2/python_api/api.html#unique) paths so that each path
       has a distinct sequence of tokens
  - 4. Compute the edit distance of each path with the ground truth
  - 5. The path with the lowest edit distance is the final output and is used to
       compute the WER

The command to compute the Nbest oracle WER is:

```bash
for epoch in 27; do
  for avg in 10 ; do
    for num_paths in 50 100 200 400; do
      for nbest_scale in 0.5 0.8 1.0; do
        ./pruned_transducer_stateless3/decode.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir ./pruned_transducer_stateless3/exp \
            --max-duration 600 \
            --decoding-method fast_beam_search_nbest_oracle \
            --num-paths $num_paths \
            --max-states 32 \
            --beam 8 \
            --nbest-scale $nbest_scale
      done
    done
  done
done
```

### LibriSpeech BPE training results (Pruned Transducer 3, 2022-05-13)

Same setup as [pruned_transducer_stateless3](./pruned_transducer_stateless3) (2022-04-29)
but change `--giga-prob` from 0.8 to 0.9. Also use `repeat` on gigaspeech XL
subset so that the gigaspeech dataloader never exhausts.

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|---------------------------------------------|
| greedy search (max sym per frame 1) | 2.03       | 4.70       | --iter 1224000 --avg 14  --max-duration 600 |
| modified beam search                | 2.00       | 4.63       | --iter 1224000 --avg 14  --max-duration 600 |
| fast beam search                    | 2.10       | 4.68       | --iter 1224000 --avg 14 --max-duration 600 |

The training commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./prepare.sh
./prepare_giga_speech.sh

./pruned_transducer_stateless3/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 0 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless3/exp-0.9 \
  --max-duration 300 \
  --use-fp16 1 \
  --lr-epochs 4 \
  --num-workers 2 \
  --giga-prob 0.9
```

The tensorboard log is available at
<https://tensorboard.dev/experiment/HpocR7dKS9KCQkJeYxfXug/>

Decoding commands:

```bash
for iter in 1224000; do
  for avg in 14; do
    for method in greedy_search modified_beam_search fast_beam_search ; do
      ./pruned_transducer_stateless3/decode.py \
        --iter $iter \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp-0.9/ \
        --max-duration 600 \
        --decoding-method $method \
        --max-sym-per-frame 1 \
        --beam 4 \
        --max-contexts 32
    done
  done
done
```

The pretrained models, training logs, decoding logs, and decoding results
can be found at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>


### LibriSpeech BPE training results (Pruned Transducer 2)

[pruned_transducer_stateless2](./pruned_transducer_stateless2)
This is with a reworked version of the conformer encoder, with many changes.

#### Training on fulll librispeech

Using commit `34aad74a2c849542dd5f6359c9e6b527e8782fd6`.
See <https://github.com/k2-fsa/icefall/pull/288>

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|-------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 2.62       | 6.37       | --epoch 25 --avg 8  --max-duration 600                                        |
| fast beam search                    | 2.61       | 6.17       | --epoch 25 --avg 8  --max-duration 600 --decoding-method fast_beam_search     |
| modified beam search                | 2.59       | 6.19       | --epoch 25 --avg 8  --max-duration 600 --decoding-method modified_beam_search |
| greedy search (max sym per frame 1) | 2.70       | 6.04       | --epoch 34 --avg 10 --max-duration 600                                        |
| fast beam search                    | 2.66       | 6.00       | --epoch 34 --avg 10  --max-duration 600 --decoding-method fast_beam_search    |
| greedy search (max sym per frame 1) | 2.62       | 6.03       | --epoch 38 --avg 10 --max-duration 600                                        |
| fast beam search                    | 2.57       | 5.95       | --epoch 38 --avg 10  --max-duration 600 --decoding-method fast_beam_search    |




The train and decode commands are:
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp --world-size 8 --num-epochs 26  --full-libri 1 --max-duration 300`
and:
`python3 ./pruned_transducer_stateless2/decode.py --exp-dir pruned_transducer_stateless2/exp --epoch 25 --avg 8 --bpe-model ./data/lang_bpe_500/bpe.model --max-duration 600`

The Tensorboard log is at <https://tensorboard.dev/experiment/Xoz0oABMTWewo1slNFXkyA> (apologies, log starts
only from epoch 3).

The pretrained models, training logs, decoding logs, and decoding results
can be found at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless2-2022-04-29>


#### Training on train-clean-100:

Trained with 1 job:
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp_100h_ws1 --world-size 1 --num-epochs 40  --full-libri 0 --max-duration 300`
and decoded with:
`python3 ./pruned_transducer_stateless2/decode.py --exp-dir pruned_transducer_stateless2/exp_100h_ws1 --epoch 19 --avg 8 --bpe-model ./data/lang_bpe_500/bpe.model --max-duration 600`.

The Tensorboard log is at <https://tensorboard.dev/experiment/AhnhooUBRPqTnaggoqo7lg> (learning rate
schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment                                               |
|-------------------------------------|------------|------------|-------------------------------------------------------|
| greedy search (max sym per frame 1) | 7.12       | 18.42      | --epoch 19 --avg 8                                    |
| greedy search (max sym per frame 1) | 6.71       | 17.77      | --epoch 29 --avg 8                                    |
| greedy search (max sym per frame 1) | 6.64       | 17.19      | --epoch 39 --avg 10                                    |
| fast beam search                    | 6.58       | 17.27      | --epoch 29 --avg 8 --decoding-method fast_beam_search |
| fast beam search                    | 6.53       | 16.82      | --epoch 39 --avg 10 --decoding-method fast_beam_search |

Trained with 2 jobs:
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp_100h_ws2 --world-size 2 --num-epochs 40  --full-libri 0 --max-duration 300`
and decoded with:
`python3 ./pruned_transducer_stateless2/decode.py --exp-dir pruned_transducer_stateless2/exp_100h_ws2 --epoch 19 --avg 8 --bpe-model ./data/lang_bpe_500/bpe.model --max-duration 600`.

The Tensorboard log is at <https://tensorboard.dev/experiment/dvOC9wsrSdWrAIdsebJILg/>
(learning rate schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.05       | 18.77      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.82       | 18.14      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.81       | 17.66      | --epoch 30  --avg 10  |


Trained with 4 jobs:
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp_100h_ws4 --world-size 4 --num-epochs 40  --full-libri 0 --max-duration 300`
and decoded with:
`python3 ./pruned_transducer_stateless2/decode.py --exp-dir pruned_transducer_stateless2/exp_100h_ws4 --epoch 19 --avg 8 --bpe-model ./data/lang_bpe_500/bpe.model --max-duration 600`.


The Tensorboard log is at <https://tensorboard.dev/experiment/a3T0TyC0R5aLj5bmFbRErA/>
(learning rate schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.31       | 19.55      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 7.08       | 18.59      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.86       | 18.29      | --epoch 30  --avg 10  |



Trained with 1 job, with  --use-fp16=True --max-duration=300 i.e. with half-precision
floats (but without increasing max-duration), after merging <https://github.com/k2-fsa/icefall/pull/305>.
Train command was
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp_100h_fp16 --world-size 1 --num-epochs 40  --full-libri 0 --max-duration 300 --use-fp16 True`

The Tensorboard log is at <https://tensorboard.dev/experiment/DAtGG9lpQJCROUDwPNxwpA>

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.10       | 18.57      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.81       | 17.84      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.63       | 17.39      | --epoch 30  --avg 10  |


Trained with 1 job, with  --use-fp16=True --max-duration=500, i.e. with half-precision
floats and max-duration increased from 300 to 500, after merging <https://github.com/k2-fsa/icefall/pull/305>.
Train command was
`python3 ./pruned_transducer_stateless2/train.py --exp-dir=pruned_transducer_stateless2/exp_100h_fp16 --world-size 1 --num-epochs 40  --full-libri 0 --max-duration 500 --use-fp16 True`

The Tensorboard log is at <https://tensorboard.dev/experiment/Km7QBHYnSLWs4qQnAJWsaA>

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.10       | 18.79      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.92       | 18.16      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.89       | 17.75      | --epoch 30  --avg 10  |




### LibriSpeech BPE training results (Pruned Transducer)

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer, a Conv1d (with kernel size 2) and a linear
layer (to transform tensor dim).

#### 2022-03-12

[pruned_transducer_stateless](./pruned_transducer_stateless)

Using commit `1603744469d167d848e074f2ea98c587153205fa`.
See <https://github.com/k2-fsa/icefall/pull/248>

The WERs are:

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| greedy search (max sym per frame 2) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| greedy search (max sym per frame 3) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| modified beam search (beam size 4)  | 2.56       | 6.27       | --epoch 42  --avg 11  --max-duration 100 |
| beam search (beam size 4)           | 2.57       | 6.27       | --epoch 42  --avg 11  --max-duration 100 |





The decoding time for `test-clean` and `test-other` is given below:
(A V100 GPU with 32 GB RAM is used for decoding. Note: Not all GPU RAM is used during decoding.)

| decoding method | test-clean (seconds) | test-other (seconds)|
|---|---:|---:|
| greedy search (--max-sym-per-frame=1) | 160 | 159 |
| greedy search (--max-sym-per-frame=2) | 184 | 177 |
| greedy search (--max-sym-per-frame=3) | 210 | 213 |
| modified beam search (--beam-size 4)| 273 | 269 |
|beam search (--beam-size 4) | 2741 | 2221 |

We recommend you to use `modified_beam_search`.

Training command:

```bash
cd egs/librispeech/ASR/
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

. path.sh

./pruned_transducer_stateless/train.py \
  --world-size 8 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 300 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/WKRFY5fYSzaVBHahenpNlA/>

The command for decoding is:

```bash
epoch=42
avg=11
sym=1

# greedy search

./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method greedy_search \
  --beam-size 4 \
  --max-sym-per-frame $sym

# modified beam search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4

# beam search
# (not recommended)
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method beam_search \
  --beam-size 4
```

You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless-2022-03-12>

#### 2022-02-18

[pruned_transducer_stateless](./pruned_transducer_stateless)


The WERs are

|                           | test-clean | test-other | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             | 2.85       | 6.98       | --epoch 28  --avg 15  --max-duration 100 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 300 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25 \
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/ejG7VpakRYePNNj6AbDEUw/#scalars>

The decoding command is:
```
epoch=28
avg=15

## greedy search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless/exp \
  --max-duration 100
```


### LibriSpeech BPE training results (Transducer)

#### Conformer encoder + embedding decoder

Conformer encoder + non-recurrent decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

See

- [./transducer_stateless](./transducer_stateless)
- [./transducer_stateless_multi_datasets](./transducer_stateless_multi_datasets)

##### 2022-03-01

Using commit `2332ba312d7ce72f08c7bac1e3312f7e3dd722dc`.

It uses [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)
as extra training data. 20% of the time it selects a batch from L subset of
GigaSpeech and 80% of the time it selects a batch from LibriSpeech.

The WERs are

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.64       | 6.55       | --epoch 39  --avg 15  --max-duration 100 |
| modified beam search (beam size 4)  | 2.61       | 6.46       | --epoch 39  --avg 15  --max-duration 100 |

The training command for reproducing is given below:

```bash
cd egs/librispeech/ASR/
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless_multi_datasets/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25 \
  --giga-prob 0.2
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/xmo5oCgrRVelH9dCeOkYBg/>

The decoding command is:

```bash
epoch=39
avg=15
sym=1

# greedy search
./transducer_stateless_multi_datasets/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --max-sym-per-frame $sym

# modified beam search
./transducer_stateless_multi_datasets/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

You can find a pretrained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-multi-datasets-bpe-500-2022-03-01>


##### 2022-04-19

[transducer_stateless2](./transducer_stateless2)

This version uses torchaudio's RNN-T loss.

Using commit `fce7f3cd9a486405ee008bcbe4999264f27774a3`.
See <https://github.com/k2-fsa/icefall/pull/316>

|                                     | test-clean | test-other | comment                                                                        |
|-------------------------------------|------------|------------|--------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 2.65       | 6.30       | --epoch 59 --avg 10  --max-duration 600                                        |
| greedy search (max sym per frame 2) | 2.62       | 6.23       | --epoch 59 --avg 10  --max-duration 100                                        |
| greedy search (max sym per frame 3) | 2.62       | 6.23       | --epoch 59 --avg 10  --max-duration 100                                        |
| modified beam search                | 2.63       | 6.15       | --epoch 59 --avg 10  --max-duration 100 --decoding-method modified_beam_search |
| beam search                         | 2.59       | 6.15       | --epoch 59 --avg 10  --max-duration 100 --decoding-method beam_search          |

**Note**: This model is trained with standard RNN-T loss. Neither modified transducer nor pruned RNN-T is used.
You can see that there is a performance degradation in WER when we limit the max symbol per frame to 1.

The number of active paths in `modified_beam_search` and `beam_search` is 4.

The training and decoding commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./transducer_stateless2/train.py \
  --world-size 8 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir transducer_stateless2/exp-2 \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5

epoch=59
avg=10
# greedy search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 600 \
  --decoding-method greedy_search \
  --max-sym-per-frame 1

# modified beam search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 100 \
  --decoding-method modified_beam_search \

# beam search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 100 \
  --decoding-method beam_search \
```

The tensorboard log is at <https://tensorboard.dev/experiment/oAlle3dxQD2EY8ePwjIGuw/>.


You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless2-torchaudio-2022-04-19>



##### 2022-02-07

Using commit `a8150021e01d34ecbd6198fe03a57eacf47a16f2`.


The WERs are

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| greedy search (max sym per frame 2) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| greedy search (max sym per frame 3) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| modified beam search (beam size 4)  | 2.67       | 6.57       | --epoch 63  --avg 19  --max-duration 100 |


The training command for reproducing is given below:

```
cd egs/librispeech/ASR/
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 76 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp-full \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/qgvWkbF2R46FYA6ZMNmOjA/#scalars>

The decoding command is:
```
epoch=63
avg=19

## greedy search
for sym in 1 2 3; do
  ./transducer_stateless/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless/exp-full \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --max-duration 100 \
    --max-sym-per-frame $sym
done

## modified beam search

./transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless/exp-full \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

You can find a pretrained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-bpe-500-2022-02-07>


#### Conformer encoder + LSTM decoder
Using commit `8187d6236c2926500da5ee854f758e621df803cc`.

Conformer encoder + LSTM decoder.

The best WER is

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 3.07       | 7.51       |

using `--epoch 34 --avg 11` with **greedy search**.

The training command to reproduce the above WER is:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer/train.py \
  --world-size 4 \
  --num-epochs 35 \
  --start-epoch 0 \
  --exp-dir transducer/exp-lr-2.5-full \
  --full-libri 1 \
  --max-duration 180 \
  --lr-factor 2.5
```

The decoding command is:

```
epoch=34
avg=11

./transducer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer/exp-lr-2.5-full \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100
```

You can find the tensorboard log at: <https://tensorboard.dev/experiment/D7NQc3xqTpyVmWi5FnWjrA>


### LibriSpeech BPE training results (Conformer-CTC)

#### 2021-11-09

The best WER, as of 2021-11-09, for the librispeech test dataset is below
(using HLG decoding + n-gram LM rescoring + attention decoder rescoring):

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.42       | 5.73       |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| 2.0            | 2.0             |


To reproduce the above result, use the following commands for training:

```
cd egs/librispeech/ASR/conformer_ctc
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc/train.py \
  --exp-dir conformer_ctc/exp_500_att0.8 \
  --lang-dir data/lang_bpe_500 \
  --att-rate 0.8 \
  --full-libri 1 \
  --max-duration 200 \
  --concatenate-cuts 0 \
  --world-size 4 \
  --bucketing-sampler 1 \
  --start-epoch 0 \
  --num-epochs 90
# Note: It trains for 90 epochs, but the best WER is at epoch-77.pt
```

and the following command for decoding

```
./conformer_ctc/decode.py \
  --exp-dir conformer_ctc/exp_500_att0.8 \
  --lang-dir data/lang_bpe_500 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 77 \
  --avg 55 \
  --method attention-decoder \
  --nbest-scale 0.5
```

You can find the pre-trained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09>

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/hZDWrZfaSqOMqtW0NEfXKg/#scalars>


#### 2021-08-19
(Wei Kang): Result of https://github.com/k2-fsa/icefall/pull/13

TensorBoard log is available at https://tensorboard.dev/experiment/GnRzq8WWQW62dK4bklXBTg/#scalars

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc

The best decoding results (WER) are listed below, we got this results by averaging models from epoch 15 to 34, and using `attention-decoder` decoder with num_paths equals to 100.

||test-clean|test-other|
|--|--|--|
|WER| 2.57% | 5.94% |

To get more unique paths, we scaled the lattice.scores with 0.5 (see https://github.com/k2-fsa/icefall/pull/10#discussion_r690951662 for more details), we searched the lm_score_scale and attention_score_scale for best results, the scales that produced the WER above are also listed below.

||lm_scale|attention_scale|
|--|--|--|
|test-clean|1.3|1.2|
|test-other|1.2|1.1|

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

# It was using ef233486, you may not need to switch to it
# git checkout ef233486

cd egs/librispeech/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python conformer_ctc/train.py --bucketing-sampler True \
                              --concatenate-cuts False \
                              --max-duration 200 \
                              --full-libri True \
                              --world-size 4 \
                              --lang-dir data/lang_bpe_5000

python conformer_ctc/decode.py --nbest-scale 0.5 \
                               --epoch 34 \
                               --avg 20 \
                               --method attention-decoder \
                               --max-duration 20 \
                               --num-paths 100 \
                               --lang-dir data/lang_bpe_5000
```

### LibriSpeech training results (Tdnn-Lstm)
#### 2021-08-24

(Wei Kang): Result of phone based Tdnn-Lstm model.

Icefall version: https://github.com/k2-fsa/icefall/commit/caa0b9e9425af27e0c6211048acb55a76ed5d315

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc

The best decoding results (WER) are listed below, we got this results by averaging models from epoch 19 to 14, and using `whole-lattice-rescoring` decoding method.

||test-clean|test-other|
|--|--|--|
|WER| 6.59% | 17.69% |

We searched the lm_score_scale for best results, the scales that produced the WER above are also listed below.

||lm_scale|
|--|--|
|test-clean|0.8|
|test-other|0.9|
