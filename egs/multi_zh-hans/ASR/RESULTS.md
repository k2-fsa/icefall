## Results

### Multi Chinese datasets char-based training results (Non-streaming) on zipformer model

This is the [pull request #1238](https://github.com/k2-fsa/icefall/pull/1238) in icefall.

#### Streaming (with CTC head)

Best results (num of params : ~69M):

The training command:

```
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --use-fp16 1 \
  --max-duration 600 \
  --num-workers 8 \
  --use-ctc 1 \
  --causal 1 \
  --exp-dir ./zipformer/exp-streaming
```

The decoding command:

```
./zipformer/decode.py \
  --causal 1 \
  --avg 1 \
  --epoch 20 \
  --chunk-size 32 \
  --left-context-frames 256 \
  --use-ctc 1 \
  --exp-dir ./zipformer/exp-streaming
```

Character Error Rates (CERs) listed below are produced by the checkpoint of the 20th epoch using BPE model ( # tokens is 2000, byte fallback enabled).

| Datasets | aidatatang _200zh | aidatatang _200zh | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech | WenetSpeech | WenetSpeech |
|--------------------------------|------------------------------|-------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|--------------------|-------------------------|---------------------|
|  Zipformer  CER   (%) | dev | test | eval | test | dev | test | dev | test | test | dev | test | dev phase1 | dev phase2 | test | dev | test meeting | test net |
| CTC Decoding | 6.17 | 6.75 | 30.0 | 31.44 | 3.8 | 4.48 | 4.69 | 5.18 | 19.59 | 10.85 | 9.83 | 14.45 | 8.53 | 16.48 | 10.24 | 9.73 | 12.03 |
| Greedy Search   | 4.71 | 5.18 | 33.78 | 35.2 | 3.28 | 3.63 | 4.34 | 4.76 | 21.99 | 4.65 | 3.98 | 10.42 | 3.94 | 12.54 | 8.16 | 9.73 | 9.66 |

Pre-trained model can be found here : https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-streaming-2023-11-05/

#### Non-streaming (with CTC head)

Best results (num of params : ~69M):

The training command:

```
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --use-fp16 1 \
  --max-duration 600 \
  --num-workers 8 \
  --use-ctc 1
```

The decoding command:

```
./zipformer/decode.py \
  --epoch 20 \
  --avg 1 \
  --use-ctc 1
```

Character Error Rates (CERs) listed below are produced by the checkpoint of the 20th epoch using BPE model ( # tokens is 2000, byte fallback enabled).

| Datasets | aidatatang _200zh | aidatatang _200zh | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech | WenetSpeech | WenetSpeech |
|--------------------------------|------------------------------|-------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|--------------------|-------------------------|---------------------|
|  Zipformer  CER   (%) | dev | test | eval | test | dev | test | dev | test | test | dev | test | dev phase1 | dev phase2 | test | dev | test meeting | test net |
| CTC Decoding | 2.86 | 3.36 | 22.93 | 24.28 | 2.05 | 2.27 | 3.33 | 3.82 | 15.45 | 3.49 | 2.77 | 6.90 | 2.85 | 8.29 | 9.41 | 6.92 | 8.57 |
| Greedy Search   | 3.36 | 3.83 | 23.90 | 25.18 | 2.77 | 3.08 | 3.70 | 4.04 | 16.13 | 3.77 | 3.15 | 6.88 | 3.14 | 8.08 | 9.04 | 7.19 | 8.17 |

Pre-trained model can be found here : https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-ctc-2023-10-24/

#### Non-streaming

Best results (num of params : ~69M):

The training command:

```
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --use-fp16 1 \
  --max-duration 600 \
  --num-workers 8
```

The decoding command:

```
./zipformer/decode.py \
  --epoch 20 \
  --avg 1
```

Character Error Rates (CERs) listed below are produced by the checkpoint of the 20th epoch using greedy search and BPE model ( # tokens is 2000, byte fallback enabled).

| Datasets | aidatatang _200zh | aidatatang _200zh | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech | WenetSpeech | WenetSpeech |
|--------------------------------|------------------------------|-------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|--------------------|-------------------------|---------------------|
| Zipformer CER   (%) | dev | test | eval| test | dev | test | dev| test | test | dev| test | dev phase1 | dev phase2 | test | dev | test meeting | test net |
| Greedy Search | 3.2 | 3.67 | 23.15 | 24.78 | 2.91 | 3.04 | 3.59 | 4.03 | 15.68 | 3.68 | 3.12 | 6.69 | 3.19 | 8.01 | 9.32 | 7.05 | 8.78 |


Pre-trained model can be found here : https://huggingface.co/zrjin/icefall-asr-multi-zh-hans-zipformer-2023-9-2/
