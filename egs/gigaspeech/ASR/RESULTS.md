## Results

### GigaSpeech BPE training results (Conformer-CTC)

#### 2022-04-06

The best WER, as of 2022-04-06, for the gigaspeech is below

Results using HLG decoding + n-gram LM rescoring + attention decoder rescoring:

|     |  Dev  | Test  |
|-----|-------|-------|
| WER | 10.47 | 10.58 |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
|      0.5       |       1.3       |


To reproduce the above result, use the following commands for training:

```
cd egs/gigaspeech/ASR
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
./conformer_ctc/train.py \
  --max-duration 120 \
  --num-workers 1 \
  --world-size 8 \
  --exp-dir conformer_ctc/exp_500 \
  --lang-dir data/lang_bpe_500
```

and the following command for decoding:

```
./conformer_ctc/decode.py \
  --epoch 18 \
  --avg 6 \
  --method attention-decoder \
  --num-paths 1000 \
  --exp-dir conformer_ctc/exp_500 \
  --lang-dir data/lang_bpe_500 \
  --max-duration 20 \
  --num-workers 1
```

Results using HLG decoding + whole lattice rescoring:

|     |  Dev  | Test  |
|-----|-------|-------|
| WER | 10.51 | 10.62 |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| lm_scale |
|----------|
|   0.2    |

To reproduce the above result, use the training commands above, and the following command for decoding:

```
./conformer_ctc/decode.py \
  --epoch 18 \
  --avg 6 \
  --method whole-lattice-rescoring \
  --num-paths 1000 \
  --exp-dir conformer_ctc/exp_500 \
  --lang-dir data/lang_bpe_500 \
  --max-duration 20 \
  --num-workers 1
```
Note: the `whole-lattice-rescoring` method is about twice as fast as the `attention-decoder` method, with slightly worse WER.

Pretrained model is available at
<https://huggingface.co/wgb14/icefall-asr-gigaspeech-conformer-ctc>

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/rz63cmJXSK2fV9GceJtZXQ/>
