## Results

### Zipformer

#### Non-streaming

The training command is:

```shell
./zipformer/train.py \
  --bilingual 1 \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 600
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search
```
Word Error Rates (WERs) listed below:

|       Datasets       | ReazonSpeech |  ReazonSpeech |     LibriSpeech    |    LibriSpeech    |
|----------------------|--------------|---------------|--------------------|-------------------|
|   Zipformer WER (%)  |     dev      |     test      |     test-clean     |    test-other     |
|     greedy_search    |     5.9      |     4.07      |        3.46        |       8.35        |
| modified_beam_search |    4.87      |     3.61      |        3.28        |       8.07        |
|   fast_beam_search   |    41.04     |    36.59      |        16.14       |       22.0        |


Character Error Rates (CERs) for Japanese listed below:
|   Decoding Method    | In-Distribution CER | JSUT | CommonVoice | TEDx  |
| :------------------: | :-----------------: | :--: | :---------: | :---: | 
|    greedy search     |        12.56        | 6.93 |    9.75     | 9.67  | 
| modified beam search |        11.59        | 6.97 |    9.55     | 9.51  | 

