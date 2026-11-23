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
  --lang-dir data/lang_bpe_500 \
  --ali-dir data/ali_500
```
and  you will get four files inside the folder `data/ali_500`:

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

### Step 3: Check your extracted alignments

There is a file `test_ali.py` in `icefall/test` that can be used to test your
alignments. It uses pre-computed alignments to modify a randomly generated
`nnet_output` and it checks that we can decode the correct transcripts
from the resulting `nnet_output`.

You should get something like the following if you run that script:

```
$ ./test/test_ali.py
['THE GOOD NATURED AUDIENCE IN PITY TO FALLEN MAJESTY SHOWED FOR ONCE GREATER DEFERENCE TO THE KING THAN TO THE MINISTER AND SUNG THE PSALM WHICH THE FORMER HAD CALLED FOR', 'THE OLD SERVANT TOLD HIM QUIETLY AS THEY CREPT BACK TO DWELL THAT THIS PASSAGE THAT LED FROM THE HUT IN THE PLEASANCE TO SHERWOOD AND THAT GEOFFREY FOR THE TIME WAS HIDING WITH THE OUTLAWS IN THE FOREST', 'FOR A WHILE SHE LAY IN HER CHAIR IN HAPPY DREAMY PLEASURE AT SUN AND BIRD AND TREE', "BUT THE ESSENCE OF LUTHER'S LECTURES IS THERE"]
['THE GOOD NATURED AUDIENCE IN PITY TO FALLEN MAJESTY SHOWED FOR ONCE GREATER DEFERENCE TO THE KING THAN TO THE MINISTER AND SUNG THE PSALM WHICH THE FORMER HAD CALLED FOR', 'THE OLD SERVANT TOLD HIM QUIETLY AS THEY CREPT BACK TO GAMEWELL THAT THIS PASSAGE WAY LED FROM THE HUT IN THE PLEASANCE TO SHERWOOD AND THAT GEOFFREY FOR THE TIME WAS HIDING WITH THE OUTLAWS IN THE FOREST', 'FOR A WHILE SHE LAY IN HER CHAIR IN HAPPY DREAMY PLEASURE AT SUN AND BIRD AND TREE', "BUT THE ESSENCE OF LUTHER'S LECTURES IS THERE"]
```

### Step 4: Use your alignments in training

Please refer to `conformer_mmi/train.py` for usage. Some useful
functions are:

- `load_alignments()`, it loads alignment saved by `conformer_ctc/ali.py`
- `convert_alignments_to_tensor()`, it converts alignments to PyTorch tensors
- `lookup_alignments()`, it returns the alignments of utterances by giving the cut ID of the utterances.
