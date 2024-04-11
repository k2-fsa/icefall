## Results

### zipformer
See <https://github.com/k2-fsa/icefall/pull/1421> for more details

[zipformer](./zipformer)

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

The evaluation command is:

```bash
python zipformer/evaluate.py \
    --epoch 32 \
    --avg 8 \
    --exp-dir zipformer/exp_at_as_full \
    --max-duration 500
```
