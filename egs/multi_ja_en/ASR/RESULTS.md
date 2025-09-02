## Results

### Zipformer

#### Non-streaming

The training command is:

```shell
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 21 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --manifest-dir data/manifests
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 21 \
    --avg 15 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method greedy_search
```

To export the model with onnx:

```shell
./zipformer/export-onnx.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 21 \
  --avg 1 \
  --exp-dir ./zipformer/exp
```

Word Error Rates (WERs) listed below:

|       Datasets       | ReazonSpeech |  ReazonSpeech |     LibriSpeech    |    LibriSpeech    |
|----------------------|--------------|---------------|--------------------|-------------------|
|   Zipformer WER (%)  |     dev      |     test      |     test-clean     |    test-other     |
|     greedy_search    |     5.9      |     4.07      |        3.46        |       8.35        |
| modified_beam_search |    4.87      |     3.61      |        3.28        |       8.07        |



We also include WER% for common English ASR datasets:

| Corpus                       | WER (%) |
|-----------------------------|---------|
| CommonVoice                 | 29.03   |
| TED                         | 16.78   |
| MLS English (test-clean)    | 8.64   |


And CER% for common Japanese datasets:

| Corpus        | CER (%) |
|---------------|---------|
| JSUT          | 8.13   |
| CommonVoice   | 9.82   |
| TEDx          | 11.64   |


Pre-trained model can be found here: [https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/m](https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k)

