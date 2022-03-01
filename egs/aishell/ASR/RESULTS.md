## Results
### Aishell training result(Transducer-stateless)
#### 2022-2-19
(Duo Ma): The tensorboard log for training is available at https://tensorboard.dev/experiment/25PmX3MxSVGTdvIdhOwllw/#scalars
You can find a pretrained model by visiting https://huggingface.co/shuanguanma/icefall_aishell_transducer_stateless_context_size2_epoch60_2022_2_19
|                           | test |comment                                  |
|---------------------------|------|-----------------------------------------|
| greedy search             | 5.4 |--epoch 59, --avg 10, --max-duration 100 |
| beam search               | 5.05|--epoch 59, --avg 10, --max-duration 100 |

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
