## Results

### Aishell training results (Transducer-stateless)
#### 2022-02-18
(Pingfeng Luo) : The tensorboard log for training is available at <https://tensorboard.dev/experiment/SG1KV62hRzO5YZswwMQnoQ/>

||test|
|--|--|
|CER| 5.4% |

You can use the following commands to reproduce our results:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
./transducer_stateless/train.py \
  --bucketing-sampler True \
  --world-size 8 \
  --lang-dir data/lang_char \
  --num-epochs 40 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp_char \
  --max-duration 160 \
  --lr-factor 3

./transducer_stateless/decode.py \
  --epoch 39 \
  --avg 10 \
  --lang-dir data/lang_char \
  --exp-dir transducer_stateless/exp_char \
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
