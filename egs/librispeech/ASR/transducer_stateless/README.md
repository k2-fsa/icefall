## Introduction

The decoder, i.e., the prediction network, is from
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419
(Rnn-Transducer with Stateless Prediction Network)

You can use the following command to start the training:

```bash
cd egs/librispeech/ASR

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 250 \
  --lr-factor 2.5
```

## How to get framewise token alignment

Assume that you already have a trained model. If not, you can either
train one by yourself or download a pre-trained model from hugging face:
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-multi-datasets-bpe-500-2022-03-01>

**Caution**: If you are going to use your own trained model, remember
to set `--modified-transducer-prob` to a nonzero value since the
force alignment code assumes that `--max-sym-per-frame` is 1.


The following shows how to get framewise token alignment using the above
pre-trained model.

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall/egs/librispeech/ASR
mkdir tmp
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-multi-datasets-bpe-500-2022-03-01 ./tmp/

ln -s $PWD/tmp/exp/pretrained.pt $PWD/tmp/epoch-999.pt

./transducer_stateless/compute_ali.py \
        --exp-dir ./tmp/exp \
        --bpe-model ./tmp/data/lang_bpe_500/bpe.model \
        --epoch 999 \
        --avg 1 \
        --max-duration 100 \
        --dataset dev-clean \
        --out-dir data/ali
```

After running the above commands, you will find the following two files
in the folder `./data/ali`:

```
-rw-r--r-- 1 xxx xxx 412K Mar  7 15:45 cuts_dev-clean.json.gz
-rw-r--r-- 1 xxx xxx 2.9M Mar  7 15:45 token_ali_dev-clean.h5
```

You can find usage examples in `./test_compute_ali.py` about
extracting framewise token alignment information from the above
two files.

## How to get word starting time from framewise token alignment

Assume you have run the above commands to get framewise token alignment
using a pre-trained model from `tmp/exp/epoch-999.pt`. You can use the following
commands to obtain word starting time.

```bash
./transducer_stateless/test_compute_ali.py \
        --bpe-model ./tmp/data/lang_bpe_500/bpe.model \
        --ali-dir data/ali \
        --dataset dev-clean
```

**Caution**: Since the frame shift is 10ms and the subsampling factor
of the model is 4, the time resolution is 0.04 second.

**Note**: The script `test_compute_ali.py` is for illustration only
and it processes only one batch and then exits.

You will get the following output:

```
5694-64029-0022-1998-0
[('THE', '0.20'), ('LEADEN', '0.36'), ('HAIL', '0.72'), ('STORM', '1.00'), ('SWEPT', '1.48'), ('THEM', '1.88'), ('OFF', '2.00'), ('THE', '2.24'), ('FIELD', '2.36'), ('THEY', '3.20'), ('FELL', '3.36'), ('BACK', '3.64'), ('AND', '3.92'), ('RE', '4.04'), ('FORMED', '4.20')]

3081-166546-0040-308-0
[('IN', '0.32'), ('OLDEN', '0.60'), ('DAYS', '1.00'), ('THEY', '1.40'), ('WOULD', '1.56'), ('HAVE', '1.76'), ('SAID', '1.92'), ('STRUCK', '2.60'), ('BY', '3.16'), ('A', '3.36'), ('BOLT', '3.44'), ('FROM', '3.84'), ('HEAVEN', '4.04')]

2035-147960-0016-1283-0
[('A', '0.44'), ('SNAKE', '0.52'), ('OF', '0.84'), ('HIS', '0.96'), ('SIZE', '1.12'), ('IN', '1.60'), ('FIGHTING', '1.72'), ('TRIM', '2.12'), ('WOULD', '2.56'), ('BE', '2.76'), ('MORE', '2.88'), ('THAN', '3.08'), ('ANY', '3.28'), ('BOY', '3.56'), ('COULD', '3.88'), ('HANDLE', '4.04')]

2428-83699-0020-1734-0
[('WHEN', '0.28'), ('THE', '0.48'), ('TRAP', '0.60'), ('DID', '0.88'), ('APPEAR', '1.08'), ('IT', '1.80'), ('LOOKED', '1.96'), ('TO',
'2.24'), ('ME', '2.36'), ('UNCOMMONLY', '2.52'), ('LIKE', '3.16'), ('AN', '3.40'), ('OPEN', '3.56'), ('SPRING', '3.92'), ('CART', '4.28')]

8297-275154-0026-2108-0
[('LET', '0.44'), ('ME', '0.72'), ('REST', '0.92'), ('A', '1.32'), ('LITTLE', '1.40'), ('HE', '1.80'), ('PLEADED', '2.00'), ('IF', '3.04'), ("I'M", '3.28'), ('NOT', '3.52'), ('IN', '3.76'), ('THE', '3.88'), ('WAY', '4.00')]

652-129742-0007-1002-0
[('SURROUND', '0.28'), ('WITH', '0.80'), ('A', '0.92'), ('GARNISH', '1.00'), ('OF', '1.44'), ('COOKED', '1.56'), ('AND', '1.88'), ('DICED', '4.16'), ('CARROTS', '4.28'), ('TURNIPS', '4.44'), ('GREEN', '4.60'), ('PEAS', '4.72')]
```


For the row:
```
5694-64029-0022-1998-0
[('THE', '0.20'), ('LEADEN', '0.36'), ('HAIL', '0.72'), ('STORM', '1.00'), ('SWEPT', '1.48'),
('THEM', '1.88'), ('OFF', '2.00'), ('THE', '2.24'), ('FIELD', '2.36'), ('THEY', '3.20'), ('FELL', '3.36'),
('BACK', '3.64'), ('AND', '3.92'), ('RE', '4.04'), ('FORMED', '4.20')]
```

- `5694-64029-0022-1998-0` is the cut ID.
- `('THE', '0.20')` means the word `THE` starts at 0.20 second.
- `('LEADEN', '0.36')` means the word `LEADEN` starts at 0.36 second.


You can compare the above word starting time with the one
from <https://github.com/CorentinJ/librispeech-alignments>

```
5694-64029-0022 ",THE,LEADEN,HAIL,STORM,SWEPT,THEM,OFF,THE,FIELD,,THEY,FELL,BACK,AND,RE,FORMED," "0.230,0.360,0.670,1.010,1.440,1.860,1.990,2.230,2.350,2.870,3.230,3.390,3.660,3.960,4.060,4.160,4.850,4.9"
```

We reformat it below for readability:

```
5694-64029-0022 ",THE,LEADEN,HAIL,STORM,SWEPT,THEM,OFF,THE,FIELD,,THEY,FELL,BACK,AND,RE,FORMED,"
"0.230,0.360,0.670,1.010,1.440,1.860,1.990,2.230,2.350,2.870,3.230,3.390,3.660,3.960,4.060,4.160,4.850,4.9"
  the  leaden hail storm swept them  off   the   field  sil   they  fell  back  and   re   formed  sil
```
