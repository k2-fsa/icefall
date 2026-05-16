# Results for mucs hi-en and bn-en

This page shows the WERs for the code switched test corpus of MUCS hi-en and bn-en.

## using conformer ctc

The following results are obtained with run.sh

Specify the language through dataset arg (hi-en or bn-en)
LM is trained using kenlm, with the training corpus

Here are the results with different decoding methods

bn-en
|                         | test  |
|-------------------------|-------|
| ctc decoding            | 31.72 |
| 1best                   | 28.05 |
| nbest                   | 27.92 |
| nbest-rescoring         | 27.22 |
| whole-lattice-rescoring | 27.24 |
| attention-decoder       | 26.46 |

hi-en
|                         | test  |
|-------------------------|-------|
| ctc decoding            | 31.43 |
| 1best                   | 28.48 |
| nbest                   | 28.55 |
| nbest-rescoring         | 28.23 |
| whole-lattice-rescoring | 28.77 |
| attention-decoder       | 28.16 |

The training command for reproducing is given below:
```bash
cd egs/mucs/ASR/
./prepare.sh

dataset="hi-en" #hi-en or bn-en
bpe=400
datadir=data_"$dataset"
./conformer_ctc/train.py \
    --num-epochs 60 \
    --max-duration 300 \
    --exp-dir ./conformer_ctc/exp_"$dataset"_bpe"$bpe" \
    --manifest-dir $datadir/fbank \
    --lang-dir $datadir/lang_bpe_"$bpe" \
    --enable-musan False \
```

The decoding command is given below:
```bash
dataset="hi-en" #hi-en or bn-en
bpe=400
datadir=data_"$dataset"
num_paths=10
max_duration=10
decode_methods="attention-decoder 1best nbest nbest-rescoring ctc-decoding whole-lattice-rescoring"

for decode_method in $decode_methods; 
do
    ./conformer_ctc/decode.py \
        --epoch 59 \
        --avg 10 \
        --manifest-dir $datadir/fbank \
        --exp-dir ./conformer_ctc/exp_"$dataset"_bpe"$bpe" \
        --max-duration $max_duration \
        --lang-dir $datadir/lang_bpe_"$bpe" \
        --lm-dir $datadir/"lm" \
        --method $decode_method \
        --num-paths $num_paths \
        
done
```