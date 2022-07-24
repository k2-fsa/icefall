## Results

### Aishell training result(Stateless Transducer)

#### Pruned transducer stateless 3

See <https://github.com/k2-fsa/icefall/pull/436>


[./pruned_transducer_stateless3](./pruned_transducer_stateless3)

It uses pruned RNN-T.

|                        | test | dev  | comment                               |
|------------------------|------|------|---------------------------------------|
| greedy search          | 5.39 | 5.09 | --epoch 29 --avg 5 --max-duration 600 |
| modified beam search   | 5.05 | 4.79 | --epoch 29 --avg 5 --max-duration 600 |
| fast beam search       | 5.13 | 4.91 | --epoch 29 --avg 5 --max-duration 600 |

Training command is:

```bash
./prepare.sh
./prepare_aidatatang_200zh.sh

export CUDA_VISIBLE_DEVICES="4,5,6,7"

./pruned_transducer_stateless3/train.py \
  --exp-dir ./pruned_transducer_stateless3/exp-context-size-1 \
  --world-size 4 \
  --max-duration 200 \
  --datatang-prob 0.5 \
  --start-epoch 1 \
  --num-epochs 30 \
  --use-fp16 1 \
  --num-encoder-layers 12 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --context-size 1 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --master-port 12356
```

**Caution**: It uses `--context-size=1`.

The tensorboard log is available at
<https://tensorboard.dev/experiment/OKKacljwR6ik7rbDr5gMqQ>

The decoding command is:

```bash
for epoch in 29; do
  for avg in 5; do
    for m in greedy_search modified_beam_search fast_beam_search; do
      ./pruned_transducer_stateless3/decode.py \
        --exp-dir ./pruned_transducer_stateless3/exp-context-size-1 \
        --epoch $epoch \
        --avg $avg \
        --use-averaged-model 1 \
        --max-duration 600 \
        --decoding-method $m \
        --num-encoder-layers 12 \
        --dim-feedforward 2048 \
        --nhead 8 \
        --context-size 1 \
        --encoder-dim 512 \
        --decoder-dim 512 \
        --joiner-dim 512
    done
  done
done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20>

We have a tutorial in [sherpa](https://github.com/k2-fsa/sherpa) about how
to use the pre-trained model for non-streaming ASR. See
<https://k2-fsa.github.io/sherpa/offline_asr/conformer/aishell.html>

#### 2022-03-01

[./transducer_stateless_modified-2](./transducer_stateless_modified-2)

It uses [optimized_transducer](https://github.com/csukuangfj/optimized_transducer)
for computing RNN-T loss.

Stateless transducer + modified transducer + using [aidatatang_200zh](http://www.openslr.org/62/) as extra training data.


|                        | test |comment                                                         |
|------------------------|------|----------------------------------------------------------------|
| greedy search          | 4.94 |--epoch 89, --avg 38, --max-duration 100, --max-sym-per-frame 1 |
| modified beam search   | 4.68 |--epoch 89, --avg 38, --max-duration 100  --beam-size 4         |

The training commands are:

```bash
cd egs/aishell/ASR
./prepare.sh --stop-stage 6
./prepare_aidatatang_200zh.sh

export CUDA_VISIBLE_DEVICES="0,1,2"

./transducer_stateless_modified-2/train.py \
  --world-size 3 \
  --num-epochs 90 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_modified-2/exp-2 \
  --max-duration 250 \
  --lr-factor 2.0 \
  --context-size 2 \
  --modified-transducer-prob 0.25 \
  --datatang-prob 0.2
```

The tensorboard log is available at
<https://tensorboard.dev/experiment/oG72ZlWaSGua6fXkcGRRjA/>

The commands for decoding are

```bash
# greedy search
for epoch in 89; do
  for avg in 38; do
  ./transducer_stateless_modified-2/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified-2/exp-2 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method greedy_search \
    --max-sym-per-frame 1
  done
done

# modified beam search
for epoch in 89; do
  for avg in 38; do
    ./transducer_stateless_modified-2/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified-2/exp-2 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method modified_beam_search \
    --beam-size 4
  done
done
```

You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2-2022-03-01>

#### 2022-03-01

[./transducer_stateless_modified](./transducer_stateless_modified)

Stateless transducer + modified transducer.

|                        | test |comment                                                         |
|------------------------|------|----------------------------------------------------------------|
| greedy search          | 5.22 |--epoch 64, --avg 33, --max-duration 100, --max-sym-per-frame 1 |
| modified beam search   | 5.02 |--epoch 64, --avg 33, --max-duration 100  --beam-size 4         |

The training commands are:

```bash
cd egs/aishell/ASR
./prepare.sh --stop-stage 6

export CUDA_VISIBLE_DEVICES="0,1,2"

./transducer_stateless_modified/train.py \
  --world-size 3 \
  --num-epochs 90 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_modified/exp-4 \
  --max-duration 250 \
  --lr-factor 2.0 \
  --context-size 2 \
  --modified-transducer-prob 0.25
```

The tensorboard log is available at
<https://tensorboard.dev/experiment/C27M8YxRQCa1t2XglTqlWg/>

The commands for decoding are

```bash
# greedy search
for epoch in 64; do
  for avg in 33; do
  ./transducer_stateless_modified/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified/exp-4 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method greedy_search \
    --max-sym-per-frame 1
  done
done

# modified beam search
for epoch in 64; do
  for avg in 33; do
    ./transducer_stateless_modified/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless_modified/exp-4 \
    --max-duration 100 \
    --context-size 2 \
    --decoding-method modified_beam_search \
    --beam-size 4
  done
done
```

You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-aishell-transducer-stateless-modified-2022-03-01>


#### 2022-2-19
(Duo Ma): The tensorboard log for training is available at https://tensorboard.dev/experiment/25PmX3MxSVGTdvIdhOwllw/#scalars
You can find a pretrained model by visiting https://huggingface.co/shuanguanma/icefall_aishell_transducer_stateless_context_size2_epoch60_2022_2_19
|                           | test |comment                                  |
|---------------------------|------|-----------------------------------------|
| greedy search             | 5.4 |--epoch 59, --avg 10, --max-duration 100  |
| beam search               | 5.05|--epoch 59, --avg 10, --max-duration 100  |

You can use the following commands to reproduce our results:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python3 ./transducer_stateless/train.py \
      --world-size 4 \
      --num-epochs 60 \
      --start-epoch 0 \
      --exp-dir exp/transducer_stateless_context_size2 \
      --max-duration 100 \
      --lr-factor 2.5 \
      --context-size 2

lang_dir=data/lang_char
dir=exp/transducer_stateless_context_size2
python3 ./transducer_stateless/decode.py \
       --epoch 59 \
       --avg 10 \
       --exp-dir $dir \
       --lang-dir $lang_dir \
       --decoding-method greedy_search \
       --context-size 2 \
       --max-sym-per-frame 3

lang_dir=data/lang_char
dir=exp/transducer_stateless_context_size2
python3 ./transducer_stateless/decode.py \
       --epoch 59 \
       --avg 10 \
       --exp-dir $dir \
       --lang-dir $lang_dir \
       --decoding-method beam_search \
       --context-size 2 \
       --max-sym-per-frame 3
```

### Aishell training results (Transducer-stateless)
#### 2022-02-18
(Pingfeng Luo) : The tensorboard log for training is available at <https://tensorboard.dev/experiment/k3QL6QMhRbCwCKYKM9po9w/>
And pretrained model is available at <https://huggingface.co/pfluo/icefall-aishell-transducer-stateless-char-2021-12-29>

||test|
|--|--|
|CER| 5.05% |

You can use the following commands to reproduce our results:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
./transducer_stateless/train.py \
  --bucketing-sampler True \
  --world-size 8 \
  --lang-dir data/lang_char \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp_rnnt_k2 \
  --max-duration 80 \
  --lr-factor 3

./transducer_stateless/decode.py \
  --epoch 59 \
  --avg 10 \
  --lang-dir data/lang_char \
  --exp-dir transducer_stateless/exp_rnnt_k2 \
  --max-duration 100 \
  --decoding-method beam_search \
  --beam-size 4
```

### Aishell training results (Conformer-MMI)
#### 2021-12-04
(Pingfeng Luo): Result of <https://github.com/k2-fsa/icefall/pull/140>

The tensorboard log for training is available at <https://tensorboard.dev/experiment/PSRYVbptRGynqpPRSykp1g>

And pretrained model is available at <https://huggingface.co/pfluo/icefall_aishell_mmi_model>

The best decoding results (CER) are listed below, we got this results by averaging models from epoch 61 to 85, and using `attention-decoder` decoder with num_paths equals to 100.

||test|
|--|--|
|CER| 4.94% |

||lm_scale|attention_scale|
|--|--|--|
|test|1.1|0.3|

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

cd egs/aishell/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
python conformer_mmi/train.py --bucketing-sampler True \
                              --max-duration 200 \
                              --start-epoch 0 \
                              --num-epochs 90 \
                              --world-size 8

python conformer_mmi/decode.py --nbest-scale 0.5 \
                               --epoch 85 \
                               --avg 25 \
                               --method attention-decoder \
                               --max-duration 20 \
                               --num-paths 100
```

### Aishell training results (Conformer-CTC)
#### 2021-11-16
(Wei Kang): Result of https://github.com/k2-fsa/icefall/pull/30

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc

The best decoding results (CER) are listed below, we got this results by averaging models from epoch 60 to 84, and using `attention-decoder` decoder with num_paths equals to 100.

||test|
|--|--|
|CER| 4.26% |

To get more unique paths, we scaled the lattice.scores with 0.5 (see https://github.com/k2-fsa/icefall/pull/10#discussion_r690951662 for more details), we searched the lm_score_scale and attention_score_scale for best results, the scales that produced the CER above are also listed below.

||lm_scale|attention_scale|
|--|--|--|
|test|0.3|0.9|

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

cd egs/aishell/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python conformer_ctc/train.py --bucketing-sampler True \
                              --max-duration 200 \
                              --start-epoch 0 \
                              --num-epochs 90 \
                              --world-size 4

python conformer_ctc/decode.py --nbest-scale 0.5 \
                               --epoch 84 \
                               --avg 25 \
                               --method attention-decoder \
                               --max-duration 20 \
                               --num-paths 100
```

### Aishell training results (Tdnn-Lstm)
#### 2021-09-13

(Wei Kang): Result of phone based Tdnn-Lstm model, https://github.com/k2-fsa/icefall/pull/30

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_aishell_conformer_ctc_lstm_ctc

The best decoding results (CER) are listed below, we got this results by averaging models from epoch 19 to 8, and using `1best` decoding method.

||test|
|--|--|
|CER| 10.16% |
