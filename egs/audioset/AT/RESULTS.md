## Results

### zipformer
See <https://github.com/k2-fsa/icefall/pull/1421> for more details

[zipformer](./zipformer)

#### normal-scaled model, number of model parameters: 65549011, i.e., 65.55 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-2024-03-12#/>

The model achieves the following mean averaged precision on AudioSet:

| Model | mAP |
| ------ | ------- |
| Zipformer-AT | 45.1 |

The training command is:

```bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
subset=full

python zipformer/train.py \
    --world-size 4 \
    --num-epochs 50 \
    --exp-dir zipformer/exp_at_as_${subset} \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --audioset-subset $subset \
    --max-duration 1000 \
    --enable-musan True \
    --master-port 13455
```

We recommend that you train the model with weighted sampler, as the model converges
faster with better performance:

| Model | mAP |
| ------ | ------- |
| Zipformer-AT, train with weighted sampler | 46.6 |

The evaluation command is:

```bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
subset=full
weighted_sampler=1
bucket_sampler=0
lr_epochs=15

python zipformer/train.py \
    --world-size 4 \
    --audioset-subset $subset \
    --num-epochs 120 \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --lr-epochs $lr_epochs \
    --exp-dir zipformer/exp_AS_${subset}_weighted_sampler${weighted_sampler} \
    --weighted-sampler $weighted_sampler \
    --bucketing-sampler $bucket_sampler \
    --max-duration 1000 \
    --enable-musan True \
    --master-port 13452
```

The command for evaluation is the same. The pre-trained model can be downloaded from https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-M-weighted-sampler


#### small-scaled model, number of model parameters: 22125218, i.e., 22.13 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/marcoyang/icefall-audio-tagging-audioset-zipformer-small-2024-04-23#/>

The model achieves the following mean averaged precision on AudioSet:

| Model | mAP |
| ------ | ------- |
| Zipformer-S-AT | 45.1 |

The training command is:

```bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
subset=full

python zipformer/train.py \
    --world-size 4 \
    --num-epochs 50 \
    --exp-dir zipformer/exp_small_at_as_${subset} \
    --start-epoch 1 \
    --use-fp16 1 \
    --num-events 527 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --audioset-subset $subset \
    --max-duration 1200 \
    --enable-musan True \
    --master-port 13455
```

The evaluation command is:

```bash
python zipformer/evaluate.py \
    --epoch 31 \
    --avg 4 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --exp-dir zipformer/exp_small_at_as_full \
    --max-duration 500
```
