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

To export the model with onnx:

```shell
./zipformer/export-onnx.py   --tokens data/lang_bbpe_2000/tokens.txt   --use-averaged-model 0   --epoch 35   --avg 1   --exp-dir zipformer/exp   --num-encoder-layers "2,2,3,4,3,2"   --downsampling-factor "1,2,4,8,4,2"   --feedforward-dim "512,768,1024,1536,1024,768"   --num-heads "4,4,4,8,4,4"   --encoder-dim "192,256,384,512,384,256"   --query-head-dim 32   --value-head-dim 12   --pos-head-dim 4   --pos-dim 48   --encoder-unmasked-dim "192,192,256,256,256,192"   --cnn-module-kernel "31,31,15,15,15,31"   --decoder-dim 512   --joiner-dim 512   --causal False   --chunk-size "16,32,64,-1"   --left-context-frames "64,128,256,-1"   --fp16 True
```
Word Error Rates (WERs) listed below:

|       Datasets       | ReazonSpeech |  ReazonSpeech |     LibriSpeech    |    LibriSpeech    |
|----------------------|--------------|---------------|--------------------|-------------------|
|   Zipformer WER (%)  |     dev      |     test      |     test-clean     |    test-other     |
|     greedy_search    |     5.9      |     4.07      |        3.46        |       8.35        |
| modified_beam_search |    4.87      |     3.61      |        3.28        |       8.07        |


Character Error Rates (CERs) for Japanese listed below:
|   Decoding Method    | In-Distribution CER | JSUT | CommonVoice | TEDx  |
| :------------------: | :-----------------: | :--: | :---------: | :---: | 
|    greedy search     |        12.56        | 6.93 |    9.75     | 9.67  | 
| modified beam search |        11.59        | 6.97 |    9.55     | 9.51  | 

Pre-trained model can be found here: https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en/tree/main

