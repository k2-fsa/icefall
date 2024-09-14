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
|     greedy_search    |    12.08     |     9.67      |        2.73        |       6.67        |
| modified_beam_search |    11.6      |     8.93      |        2.69        |       6.52        |
|   fast_beam_search   |    35.38     |    37.79      |        3.06        |       7.06        |
