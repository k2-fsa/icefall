## Results
### Switchboard BPE training results (Conformer-CTC)

#### 2023-09-04

The best WER, as of 2023-09-04, for the Switchboard is below

Results using attention decoder are given as:

|                                |  eval2000-swbd  |  eval2000-callhome  | eval2000-avg |
|--------------------------------|-----------------|---------------------|--------------|
|         `conformer_ctc`        |      9.48       |         17.73       |    13.67     | 

Decoding results and models can be found here:
https://huggingface.co/zrjin/icefall-asr-swbd-conformer-ctc-2023-8-26
#### 2023-06-27

The best WER, as of 2023-06-27, for the Switchboard is below

Results using HLG decoding + n-gram LM rescoring + attention decoder rescoring:

|                                |  eval2000  |  rt03  |
|--------------------------------|------------|--------|
|         `conformer_ctc`        |    30.80   |  32.29 |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:

##### eval2000

| ngram_lm_scale | attention_scale |
|----------------|-----------------|
|      0.9       |       1.1       |

##### rt03

| ngram_lm_scale | attention_scale |
|----------------|-----------------|
|      0.9       |       1.9       |

To reproduce the above result, use the following commands for training:

```bash
cd egs/swbd/ASR
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1"
./conformer_ctc/train.py \
  --max-duration 120 \
  --num-workers 8 \
  --enable-musan False \
  --world-size 2 \
  --num-epochs 100
```

and the following command for decoding:

```bash
./conformer_ctc/decode.py \
  --epoch 99 \
  --avg 10 \
  --max-duration 50
```

#### 2023-06-26

The best WER, as of 2023-06-26, for the Switchboard is below

Results using HLG decoding + n-gram LM rescoring + attention decoder rescoring:

|                                |  eval2000  |  rt03  |
|--------------------------------|------------|--------|
|         `conformer_ctc`        |    33.37   |  35.06 |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:

##### eval2000

| ngram_lm_scale | attention_scale |
|----------------|-----------------|
|      0.3       |       2.5       |

##### rt03

| ngram_lm_scale | attention_scale |
|----------------|-----------------|
|      0.7       |       1.3       |

To reproduce the above result, use the following commands for training:

```bash
cd egs/swbd/ASR
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1"
./conformer_ctc/train.py \
  --max-duration 120 \
  --num-workers 8 \
  --enable-musan False \
  --world-size 2 \
```

and the following command for decoding:

```bash
./conformer_ctc/decode.py \
  --epoch 55 \
  --avg 1 \
  --max-duration 50
```

For your reference, the nbest oracle WERs are:

|                                |  eval2000  |  rt03  |
|--------------------------------|------------|--------|
|         `conformer_ctc`        |    25.64   |  26.84 |
