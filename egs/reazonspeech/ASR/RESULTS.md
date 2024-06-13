## Results

### Zipformer

#### Non-streaming

##### large-scaled model, number of model parameters: 159337842, i.e., 159.34 M

|   decoding method    | In-Distribution CER | JSUT | CommonVoice | TEDx  |      comment       |
| :------------------: | :-----------------: | :--: | :---------: | :---: | :----------------: |
|    greedy search     |         4.2         | 6.7  |    7.84     | 17.9  | --epoch 39 --avg 7 |
| modified beam search |        4.13         | 6.77 |    7.69     | 17.82 | --epoch 39 --avg 7 |

The training command is:

```shell
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --causal 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --lang data/lang_char \
  --max-duration 1600 
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 40 \
    --avg 16 \
    --exp-dir zipformer/exp-large \
    --max-duration 600 \
    --causal 0 \
    --decoding-method greedy_search \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --lang data/lang_char \
    --blank-penalty 0
```

