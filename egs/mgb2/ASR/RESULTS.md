# Results


### MGB2 all data BPE training results (Stateless Pruned Transducer)

#### 2022-09-07

The WERs are

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
|          greedy search             | 15.52      | 15.28      | --epoch 18, --avg 5, --max-duration 200 |
| modified beam search               | 13.88      | 13.7       | --epoch 18, --avg 5, --max-duration 200 |
| fast beam search                   | 14.62      | 14.36      | --epoch 18, --avg 5, --max-duration 200|

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"


  
./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless5/exp \
  --max-duration 300 \
  --num-buckets 50
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/YyNv45pfQ0GqWzZ898WOlw/#scalars

The decoding command is:
```
epoch=18
avg=5
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch $epoch \
	--beam-size 10 \
    --avg $avg \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 200 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 12 \
    --dim-feedforward 2048 \
    --nhead 8 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-averaged-model True
done
```

### MGB2 all data BPE training results (Conformer-CTC) (after 40 epochs)

#### 2022-06-04

You can find a pretrained model, training logs, decoding logs, and decoding results at:
https://huggingface.co/AmirHussein/icefall-asr-mgb2-conformer_ctc-2022-27-06

The best WER, as of 2022-06-04, for the MGB2 test dataset is below

Using whole lattice HLG decoding + n-gram LM rescoring 

|     | dev        | test       |
|-----|------------|------------|
| WER | 15.62      |  15.01     |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| 0.1            | -            |


Using n-best (n=0.5) attention decoder rescoring

|     | dev        | test       |
|-----|------------|------------|
| WER |    15.89   |  15.08     |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:
| ngram_lm_scale | attention_scale |
|----------------|-----------------|
| 0.01           | 0.5             |


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
  --num-epochs 40
  
```

and the following command for nbest decoding

```
./conformer_ctc/decode.py \
  --lang-dir data/lang_bpe_5000 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 40 \
  --avg 5 \
  --method attention-decoder \
  --nbest-scale 0.5
```

and the following command for whole-lattice decoding

```
./conformer_ctc/decode.py \
  --epoch 40 \
  --avg 5 \
  --exp-dir conformer_ctc/exp_5000_att0.8 \
  --lang-dir data/lang_bpe_5000 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --method  whole-lattice-rescoring
```


The tensorboard log for training is available at
https://tensorboard.dev/experiment/QYNzOi52RwOX8yvtpl3hMw/#scalars


### MGB2 100h BPE training results (Conformer-CTC) (after 33 epochs)

#### 2022-06-04

The best WER, as of 2022-06-04, for the MGB2 test dataset is below

Using whole lattice HLG decoding + n-gram LM rescoring 

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
  --num-epochs 40
  
```

and the following command for nbest decoding

```
./conformer_ctc/decode.py \
  --lang-dir data/lang_bpe_5000 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 40 \
  --avg 5 \
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
  --epoch 40 \
  --avg 5 \
  --method  whole-lattice-rescoring
```

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/zy6FnumCQlmiO7BPsdCmEg/#scalars>




