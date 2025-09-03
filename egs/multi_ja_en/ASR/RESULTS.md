## Results

### Zipformer

#### Non-streaming (Byte-Level BPE vocab_size=2000)

Trained on 15k hours of ReazonSpeech (filtered to only audio segments between 8s and 22s) and 15k hours of MLS English.

The training command is:

```shell
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 10 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --manifest-dir data/manifests \
  --enable-musan True
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 10 \
    --avg 1 \
    --exp-dir ./zipformer/exp \
    --decoding-method modified_beam_search \
    --manifest-dir data/manifests
```

To export the model with onnx:

```shell
./zipformer/export-onnx.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 10 \
  --avg 1 \
  --exp-dir ./zipformer/exp
```

WER and CER on test set listed below (calculated with `./zipformer/decode.py`):

|       Datasets       | ReazonSpeech + MLS English (combined test set) |
|----------------------|------------------------------------------------|
|   Zipformer WER (%)  |                      test                      |
|     greedy_search    |                      6.33                      |
| modified_beam_search |                      6.32                      |



We also include WER% for common English ASR datasets:

| Corpus                      | WER (%) |
|-----------------------------|---------|
| CommonVoice                 | 29.03   |
| TED                         | 16.78   |
| MLS English (test set)      | 8.64    |


And CER% for common Japanese datasets:

| Corpus        | CER (%) |
|---------------|---------|
| JSUT          | 8.13   |
| CommonVoice   | 9.82   |
| TEDx          | 11.64   |


Pre-trained model can be found here: [https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k](https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k)

(Not yet publicly released)

#### Streaming (Byte-Level BPE vocab_size=2000)

Trained on 15k hours of ReazonSpeech (filtered to only audio segments between 8s and 22s) and 15k hours of MLS English.

The training command is:

```shell
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 10 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --manifest-dir data/manifests \
  --enable-musan True
```

The decoding command is:

```shell
./zipformer/decode.py \
    --epoch 10 \
    --avg 1 \
    --exp-dir ./zipformer/exp \
    --decoding-method modified_beam_search \
    --manifest-dir data/manifests
```

To export the model with onnx:

```shell
./zipformer/export-onnx.py \
  --tokens ./data/lang/bbpe_2000/tokens.txt \
  --use-averaged-model 0 \
  --epoch 10 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir ./zipformer/exp
```

You may also use decode chunk sizes `16`, `32`, `64`, `128`.

Word Error Rates (WERs) listed below:

*Please let us know which script to use to evaluate the streaming model!*


We also include WER% for common English ASR datasets:

*Please let us know which script to use to evaluate the streaming model!*


And CER% for common Japanese datasets:

*Please let us know which script to use to evaluate the streaming model!*


Pre-trained model can be found here: [https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k](https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/multi_ja_en_15k15k)

(Not yet publicly released)
