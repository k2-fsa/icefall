## Results

### Multi Chinese datasets char-based training results (Non-streaming) on zipformer model

This is the [pull request #1238](https://github.com/k2-fsa/icefall/pull/1238) in icefall.

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
