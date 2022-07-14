# Results

### MGB2 BPE training results (Conformer-CTC) (after 3 epochs)

#### 2022-06-04

The best WER, as of 2022-06-04, for the MGB2 test dataset is below

Using whole lattice HLG decoding + n-gram LM rescoring + attention decoder rescoring

|     | dev        | test       |
|-----|------------|------------|
| WER | 25.32      |  23.53     |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| 0.1            | -            |


Using n-best (n=0.5) HLG decoding + n-gram LM rescoring + attention decoder rescoring:

|     | dev        | test       |
|-----|------------|------------|
| WER |    27.87   |  26.12     |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| 0.01           | 0.3             |


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

and the following command for nbest decoding

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

and the following command for whole-lattice decoding

```
./conformer_ctc/decode.py \
  --lang-dir data/lang_bpe_5000 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 2 \
  --avg 2 \
  --method  whole-lattice-rescoring
```

You can find the pre-trained model by visiting
<comming soon>

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/zy6FnumCQlmiO7BPsdCmEg/#scalars>