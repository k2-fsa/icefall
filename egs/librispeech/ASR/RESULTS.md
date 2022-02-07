## Results

### LibriSpeech BPE training results (Transducer)

#### Conformer encoder + embedding decoder

Using commit `a8150021e01d34ecbd6198fe03a57eacf47a16f2`.

Conformer encoder + non-recurrent decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

The WERs are

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.68       | 6.71       | --epoch 61, --avg 18, --max-duration 100 |
| greedy search (max sym per frame 2) | 2.69       | 6.71       | --epoch 61, --avg 18, --max-duration 100 |
| greedy search (max sym per frame 3) | 2.69       | 6.71       | --epoch 61, --avg 18, --max-duration 100 |
| modified beam search (beam size 4)  | 2.67       | 6.64       | --epoch 61, --avg 18, --max-duration 100 |


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
epoch=61
avg=18

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
