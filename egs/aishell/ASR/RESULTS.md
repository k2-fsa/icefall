## Results

### Aishell training result(Stateless Transducer)

#### Pruned transducer stateless 7

[./pruned_transducer_stateless7](./pruned_transducer_stateless7)

It's Zipformer with Pruned RNNT loss.

|                        | test | dev  | comment                               |
|------------------------|------|------|---------------------------------------|
| greedy search          | 5.02 | 4.61 | --epoch 42 --avg 6 --max-duration 600 |
| modified beam search   | 4.81 | 4.4 | --epoch 42 --avg 6 --max-duration 600 |
| fast beam search       | 4.91 | 4.52 | --epoch 42 --avg 6 --max-duration 600 |

Training command is:

```bash
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1"

./pruned_transducer_stateless7/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7/exp \
  --context-size 1 \
  --max-duration 300
```

**Caution**: It uses `--context-size=1`.

The tensorboard log is available at
<https://tensorboard.dev/experiment/MHYo3ApfQxaCdYLr38cQOQ>

The decoding command is:
```bash
for m in greedy_search modified_beam_search fast_beam_search ; do
  ./pruned_transducer_stateless7/decode.py \
    --epoch 42 \
    --avg 6 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --lang-dir data/lang_char \
    --max-duration 300 \
    --context-size 1 \
    --decoding-method $m

done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/marcoyang/icefall-asr-aishell-zipformer-pruned-transducer-stateless7-2023-03-21>
#### Pruned transducer stateless 7 (zipformer)

See <https://github.com/k2-fsa/icefall/pull/986>

[./pruned_transducer_stateless7_bbpe](./pruned_transducer_stateless7_bbpe)

**Note**: The modeling units are byte level BPEs

The best results I have gotten are:

Vocab size | Greedy search(dev & test) | Modified beam search(dev & test) | Fast beam search (dev & test)  | Fast beam search LG (dev & test) | comments
-- | -- | -- | -- | -- | --
500 | 4.31 & 4.59 | 4.25 & 4.54 | 4.27 & 4.55 |  4.07 & 4.38 | --epoch 48 --avg 29

The training command:

```
export CUDA_VISIBLE_DEVICES="4,5,6,7"

./pruned_transducer_stateless7_bbpe/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 800 \
  --bpe-model data/lang_bbpe_500/bbpe.model \
  --exp-dir pruned_transducer_stateless7_bbpe/exp \
  --lr-epochs 6 \
  --master-port 12535
```

The decoding command:

```
for m in greedy_search modified_beam_search fast_beam_search fast_beam_search_LG; do
    ./pruned_transducer_stateless7_bbpe/decode.py \
      --epoch 48 \
      --avg 29 \
      --exp-dir ./pruned_transducer_stateless7_bbpe/exp \
      --max-sym-per-frame 1 \
      --ngram-lm-scale 0.25 \
      --ilme-scale 0.2 \
      --bpe-model data/lang_bbpe_500/bbpe.model \
      --max-duration 2000 \
      --decoding-method $m
done
```

The pretrained model is available at: https://huggingface.co/pkufool/icefall_asr_aishell_pruned_transducer_stateless7_bbpe


#### Pruned transducer stateless 3

See <https://github.com/k2-fsa/icefall/pull/436>


[./pruned_transducer_stateless3](./pruned_transducer_stateless3)

It uses pruned RNN-T.

|                        | test | dev  | comment                               |
|------------------------|------|------|---------------------------------------|
| greedy search          | 5.39 | 5.09 | --epoch 29 --avg 5 --max-duration 600 |
| modified beam search   | 5.05 | 4.79 | --epoch 29 --avg 5 --max-duration 600 |
| modified beam search + RNNLM shallow fusion   | 4.73 | 4.53 | --epoch 29 --avg 5 --max-duration 600 |
| modified beam search + LODR   | 4.57 | 4.37 | --epoch 29 --avg 5 --max-duration 600 |
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

We provide the option of shallow fusion with a RNN language model. The pre-trained language model is
available at <https://huggingface.co/marcoyang/icefall-aishell-rnn-lm>. To decode with the language model,
please use the following command:

```bash
# download pre-trained model
git lfs install
git clone https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20

aishell_exp=icefall-aishell-pruned-transducer-stateless3-2022-06-20/

pushd ${aishell_exp}/exp
ln -s pretrained-epoch-29-avg-5-torch-1.10.0.pt epoch-99.pt
popd

# download RNN LM
git lfs install
git clone https://huggingface.co/marcoyang/icefall-aishell-rnn-lm
rnnlm_dir=icefall-aishell-rnn-lm

# RNNLM shallow fusion
for lm_scale in $(seq 0.26 0.02 0.34); do
  python ./pruned_transducer_stateless3/decode.py \
      --epoch 99 \
      --avg 1 \
      --lang-dir ${aishell_exp}/data/lang_char \
      --exp-dir ${aishell_exp}/exp \
      --use-averaged-model False \
      --decoding-method modified_beam_search_lm_shallow_fusion \
      --use-shallow-fusion 1 \
      --lm-type rnn \
      --lm-exp-dir ${rnnlm_dir}/exp \
      --lm-epoch 99 \
      --lm-scale $lm_scale \
      --lm-avg 1 \
      --rnn-lm-embedding-dim 2048 \
      --rnn-lm-hidden-dim 2048 \
      --rnn-lm-num-layers 2 \
      --lm-vocab-size 4336
done

# RNNLM Low-order density ratio (LODR) with a 2-gram

cp ${rnnlm_dir}/2gram.fst.txt ${aishell_exp}/data/lang_char/2gram.fst.txt

for lm_scale in 0.48; do
  for LODR_scale in -0.28; do
    python ./pruned_transducer_stateless3/decode.py \
        --epoch 99 \
        --avg 1 \
        --lang-dir ${aishell_exp}/data/lang_char \
        --exp-dir ${aishell_exp}/exp \
        --use-averaged-model False \
        --decoding-method modified_beam_search_LODR \
        --use-shallow-fusion 1 \
        --lm-type rnn \
        --lm-exp-dir ${rnnlm_dir}/exp \
        --lm-epoch 99 \
        --lm-scale $lm_scale \
        --lm-avg 1 \
        --rnn-lm-embedding-dim 2048 \
        --rnn-lm-hidden-dim 2048 \
        --rnn-lm-num-layers 2 \
        --lm-vocab-size 4336 \
        --tokens-ngram 2 \
        --backoff-id 4336 \
        --ngram-lm-scale $LODR_scale
  done
done

```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/csukuangfj/icefall-aishell-pruned-transducer-stateless3-2022-06-20>

We have a tutorial in [sherpa](https://github.com/k2-fsa/sherpa) about how
to use the pre-trained model for non-streaming ASR. See
<https://k2-fsa.github.io/sherpa/offline_asr/conformer/aishell.html>


#### Pruned transducer stateless 2

See https://github.com/k2-fsa/icefall/pull/536

[./pruned_transducer_stateless2](./pruned_transducer_stateless2)

It uses pruned RNN-T.

|                      | test | dev  | comment                                |
| -------------------- | ---- | ---- | -------------------------------------- |
| greedy search        | 5.20 | 4.78 | --epoch 72 --avg 14 --max-duration 200 |
| modified beam search | 5.07 | 4.63 | --epoch 72 --avg 14 --max-duration 200 |
| fast beam search     | 5.13 | 4.70 | --epoch 72 --avg 14 --max-duration 200 |

Training command is:

```bash
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1"

./pruned_transducer_stateless2/train.py \
        --world-size 2 \
        --num-epochs 90 \
        --start-epoch 0 \
        --exp-dir pruned_transducer_stateless2/exp \
        --max-duration 200 \
```

The tensorboard log is available at
https://tensorboard.dev/experiment/QI3PVzrGRrebxpbWUPwmkA/

The decoding command is:
```bash
for m in greedy_search modified_beam_search fast_beam_search ; do
  ./pruned_transducer_stateless2/decode.py \
    --epoch 72 \
    --avg 14 \
    --exp-dir ./pruned_transducer_stateless2/exp \
    --lang-dir data/lang_char \
    --max-duration 200 \
    --decoding-method $m

done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/teapoly/icefall-aishell-pruned-transducer-stateless2-2022-08-18>


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
