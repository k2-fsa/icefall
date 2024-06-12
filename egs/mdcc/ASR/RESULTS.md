## Results

#### Zipformer

See <https://github.com/k2-fsa/icefall/pull/1537>

[./zipformer](./zipformer)

##### normal-scaled model, number of model parameters: 74470867, i.e., 74.47 M

|                        | test | valid | comment                                 |
|------------------------|------|-------|-----------------------------------------|
| greedy search          | 7.45 | 7.51  | --epoch 45 --avg 35                     |
| modified beam search   | 6.68 | 6.73  | --epoch 45 --avg 35                     |
| fast beam search       | 7.22 | 7.28  | --epoch 45 --avg 35                     |

The training command:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer/train.py \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 50 \
  --use-fp16 1 \
  --exp-dir ./zipformer/exp \
  --max-duration 1000 
```

The decoding command:

```
 ./zipformer/decode.py \
   --epoch 45 \
   --avg 35 \
   --exp-dir ./zipformer/exp \
   --decoding-method greedy_search  # modified_beam_search
```

The pretrained model is available at:  https://huggingface.co/zrjin/icefall-asr-mdcc-zipformer-2024-03-11/