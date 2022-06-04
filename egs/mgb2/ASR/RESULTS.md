# Results

### MGB2 BPE training results (Conformer-CTC)

#### 2022-06-04

The best WER, as of 2022-06-04, for the MGB2 test dataset is below
(using HLG decoding + n-gram LM rescoring + attention decoder rescoring):

|     | dev | test |
|-----|------------|------------|
| WER | -       | -      |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| -           | -            |


To reproduce the above result, use the following commands for training:

# Note: the model was trained on V-100 32GB GPU

```
cd egs/mgb2/ASR
. ./path.sh
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1"
./conformer_ctc/train.py \
  --lang-dir data/lang_bpe_5000 \
  --att-rate 0.8 \
  --lr-factor 10 \
  --max-duration  \
  --concatenate-cuts 0 \
  --world-size 2 \
  --bucketing-sampler 1 \
  --max-duration 100 \ 
  --start-epoch 0 \
  --num-epochs 30
  
```

and the following command for decoding

```
./conformer_ctc/decode.py \
  --lang-dir data/lang_bpe_5000 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 2 \
  --avg 2 \
  --method attention-decoder \
  --nbest-scale 0.5
```

You can find the pre-trained model by visiting
<comming soon>

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/zy6FnumCQlmiO7BPsdCmEg/#scalars>