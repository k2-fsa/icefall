## Introduction

Please visit
<https://icefall.readthedocs.io/en/latest/recipes/librispeech/conformer_ctc.html>
for how to run this recipe.

## How to compute framewise alignment information

### Step 1: Train a model

Please use `conformer_ctc/train.py` to train a model.
See <https://icefall.readthedocs.io/en/latest/recipes/librispeech/conformer_ctc.html>
for how to do it.

### Step 2: Compute framewise alignment

Run

```
# Choose a checkpoint and determine the number of checkpoints to average
epoch=30
avg=15
./conformer_ctc/ali.py \
  --epoch $epoch \
  --avg $avg \
  --max-duration 500 \
  --bucketing-sampler 0 \
  --full-libri 1 \
  --exp-dir conformer_ctc/exp \
  --lang-dir data/lang_bpe_5000 \
  --ali-dir data/ali_5000
```
and  you will get four files inside the folder `data/ali_5000`:

```
$ ls -lh data/ali_500
total 546M
-rw-r--r-- 1 kuangfangjun root 1.1M Sep 28 08:06 test_clean.pt
-rw-r--r-- 1 kuangfangjun root 1.1M Sep 28 08:07 test_other.pt
-rw-r--r-- 1 kuangfangjun root 542M Sep 28 11:36 train-960.pt
-rw-r--r-- 1 kuangfangjun root 2.1M Sep 28 11:38 valid.pt
```

**Note**: It can take more than 3 hours to compute the alignment
for the training dataset, which contains 960 * 3 = 2880 hours of data.

**Caution**: The model parameters in `conformer_ctc/ali.py` have to match those
in `conformer_ctc/train.py`.

**Caution**: You have to set the parameter `preserve_id` to `True` for `CutMix`.
Search `./conformer_ctc/asr_datamodule.py` for `preserve_id`.

**TODO:** Add doc about how to use the extracted alignment in the other pull-request.
