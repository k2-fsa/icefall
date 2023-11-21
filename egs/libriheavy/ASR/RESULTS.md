# Results

## zipformer (zipformer + pruned stateless transducer)

See <https://github.com/k2-fsa/icefall/pull/1261> for more details.

[zipformer](./zipformer)

### Non-streaming

#### Training on normalized text, i.e. Upper case without punctuation

##### normal-scaled model, number of model parameters: 65805511, i.e., 65.81 M

You can find a pretrained model, training logs at:
<https://www.modelscope.cn/models/pkufool/icefall-asr-zipformer-libriheavy-20230926/summary>

Note: The repository above contains three models trained on different subset of libriheavy exp(large set), exp_medium_subset(medium set),
exp_small_subset(small set).

Results of models:

| training set  |  decoding method    | librispeech clean | librispeech other | libriheavy clean | libriheavy other | comment            |
|---------------|---------------------|-------------------|-------------------|------------------|------------------|--------------------|
| small         |  greedy search      | 4.19              | 9.99              | 4.75             | 10.25            |--epoch 90 --avg 20 |
| small         | modified beam search| 4.05              | 9.89              | 4.68             | 10.01            |--epoch 90 --avg 20 |
| medium        |  greedy search      | 2.39              | 4.85              | 2.90             | 6.6              |--epoch 60 --avg 20 |
| medium        | modified beam search| 2.35              | 4.82              | 2.90             | 6.57             |--epoch 60 --avg 20 |
| large         |  greedy search      | 1.67              | 3.32              | 2.24             | 5.61             |--epoch 16 --avg 3  |
| large         | modified beam search| 1.62              | 3.36              | 2.20             | 5.57             |--epoch 16 --avg 3  |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./zipformer/train.py \
    --world-size 4 \
    --master-port 12365 \
    --exp-dir zipformer/exp \
    --num-epochs 60 \   # 16 for large; 90 for small
    --lr-hours 15000 \  # 20000 for large; 5000 for small
    --use-fp16 1 \
    --start-epoch 1 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --max-duration 1000 \
    --subset medium
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search; do
  ./zipformer/decode.py \
      --epoch 16 \
      --avg 3 \
      --exp-dir zipformer/exp \
      --max-duration 1000 \
      --causal 0 \
      --decoding-method $m
done
```

#### Training on full formatted text, i.e. with casing and punctuation

##### normal-scaled model, number of model parameters: 66074067 , i.e., 66M

You can find a pretrained model, training logs at:
<https://www.modelscope.cn/models/pkufool/icefall-asr-zipformer-libriheavy-punc-20230830/summary>

Note: The repository above contains three models trained on different subset of libriheavy exp(large set), exp_medium_subset(medium set),
exp_small_subset(small set).

Results of models:

| training set  |  decoding method    | libriheavy clean (WER) | libriheavy other (WER) | libriheavy clean (CER) | libriheavy other (CER) | comment            |
|---------------|---------------------|-------------------|-------------------|------------------|------------------|--------------------|
| small         | modified beam search| 13.04             | 19.54             | 4.51             | 7.90             |--epoch 88 --avg 41 |
| medium        | modified beam search| 9.84              | 13.39             | 3.02             | 5.10             |--epoch 50 --avg 15 |
| large         | modified beam search| 7.76              | 11.32             | 2.41             | 4.22             |--epoch 16 --avg 2  |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./zipformer/train.py \
    --world-size 4 \
    --master-port 12365 \
    --exp-dir zipformer/exp \
    --num-epochs 60 \   # 16 for large; 90 for small
    --lr-hours 15000 \  # 20000 for large; 10000 for small
    --use-fp16 1 \
    --train-with-punctuation 1 \
    --start-epoch 1 \
    --bpe-model data/lang_punc_bpe_756/bpe.model \
    --max-duration 1000 \
    --subset medium
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search; do
  ./zipformer/decode.py \
      --epoch 16 \
      --avg 3 \
      --exp-dir zipformer/exp \
      --max-duration 1000 \
      --causal 0 \
      --decoding-method $m
done
```
