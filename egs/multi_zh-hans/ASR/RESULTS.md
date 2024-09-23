## Results

### Multi Chinese datasets (without datatang 200h) finetuning results on Whisper-large-v2
#### Whisper
[./whisper](./whisper)

Character Error Rates (CERs) listed below are produced by the checkpoint of the second epoch using greedy search.

|Model| Datasets | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech  |
|-|--------------------------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|-------------------|
| | Split |           eval| test | dev | test | dev| test | test      | dev| test | dev phase1 | dev phase2 | test | test meeting |
|whisper-large-v2-ft |Greedy Search |  23.22 | 28.24 | 0.61 | 0.66 | 2.67 | 2.80 | 16.61 | 2.56 | 2.21 | 4.73 | 1.90 | 5.98 |                    8.13 |
|whisper-large-v2-ft-distill |Greedy Search | 24.91  | 26.73 | 0.91 | 0.94 | 2.71 | 2.98 | 17.65 | 2.81 | 2.47 | 5.16 | 2.10 | 6.27 |   8.34                  |

Command for training is:
```bash
pip install -r whisper/requirements.txt

# We updated the label of wenetspeech to remove OCR deletion errors, see https://github.com/wenet-e2e/WenetSpeech/discussions/54

torchrun --nproc-per-node 8 ./whisper/train.py \
  --max-duration 200 \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --deepspeed \
  --deepspeed_config ./whisper/ds_config_zero1.json
```

Command for decoding using fine-tuned models:
```bash
git lfs install
git clone https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper
ln -s icefall_asr_multi-hans-zh_whisper/v1.1/epoch-3-avg-10.pt whisper/exp_large_v2/epoch-999.pt

python3 ./whisper/decode.py \
  --exp-dir whisper/exp_large_v2 \
  --model-name large-v2 \
  --epoch 999 --avg 1 \
  --beam-size 10 --max-duration 50
```

Fine-tuned models, training logs, decoding logs, tensorboard and decoding results
are available at
<https://huggingface.co/yuekai/icefall_asr_multi-hans-zh_whisper>

### Multi Chinese datasets char-based training results (streaming) on zipformer-xl model

#### Streaming (with CTC head)

The training command for extra-large model (num of params : ~700M):

Please use the [script](https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/prepare.sh) to prepare fbank features.

```
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 20 \
  --use-fp16 1 \
  --max-duration 1200 \
  --num-workers 8 \
  --use-ctc 1 \
  --exp-dir zipformer/exp-xl \
  --causal 1 \
  --num-encoder-layers 2,3,5,6,5,3 \
  --feedforward-dim 1536,2048,3072,4096,3072,1536 \
  --encoder-dim 512,768,1024,1536,1024,512 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --decoder-dim 768 --joiner-dim 768 \
  --value-head-dim 18 \
  --query-head-dim 48 \
  --num-heads 4,4,4,8,4,4

```

The decoding command for transducer greedy search:

```
./zipformer/decode.py \
  --epoch 999 \
  --avg 1 \
  --causal 1 \
  --use-averaged-model False \
  --chunk_size -1
  --left-context-frames -1 \
  --use-ctc 1 \
  --exp-dir zipformer/exp-xl \
  --max-duration 1200 \
  --num-encoder-layers 2,3,5,6,5,3 \
  --feedforward-dim 1536,2048,3072,4096,3072,1536 \
  --encoder-dim 512,768,1024,1536,1024,512 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --decoder-dim 768 --joiner-dim 768 \
  --value-head-dim 18 \
  --query-head-dim 48 \
  --num-heads 4,4,4,8,4,4
```

Character Error Rates (CERs) listed below are produced by the checkpoint of the 18th epoch using BPE model ( # tokens is 2000, byte fallback enabled).

| Datasets | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech | WenetSpeech | WenetSpeech |
|--------------------------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|--------------------|-------------------------|---------------------|
|  Zipformer  CER   (%) |  eval | test | dev | test | dev | test | test | dev | test | dev phase1 | dev phase2 | test | dev | test meeting | test net |
| Transducer Greedy Offline   | 21.67  | 23.43 | 1.22 | 1.31 | 3.17 | 3.27 | 14.64 | 2.42 | 1.99 | 5.00 | 2.29 | 5.98 | 5.15 | 5.85 | 6.89 |

Pre-trained model can be found here : https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-xl
### Multi Chinese datasets char-based training results (streaming) on zipformer large model

#### Streaming (with CTC head)

The training command for large model (num of params : ~160M):

Please use the [script](https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/prepare.sh) to prepare fbank features.

```
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 20 \
  --use-fp16 1 \
  --max-duration 1200 \
  --num-workers 8 \
  --use-ctc 1 \
  --exp-dir zipformer/exp-large \
  --causal 1 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 768,1024,1536,2048,1536,768 \
  --encoder-dim 256,384,512,768,512,256 \
  --blank-penalty 0.7 \
  --encoder-unmasked-dim 192,192,256,320,256,192

```

The decoding command for transducer greedy search:

```
./zipformer/decode.py \
  --epoch 999 \
  --avg 1 \
  --causal 1 \
  --use-averaged-model False \
  --chunk_size -1
  --left-context-frames -1 \
  --use-ctc 1 \
  --exp-dir zipformer/exp-large \
  --max-duration 1200 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 768,1024,1536,2048,1536,768 \
  --encoder-dim 256,384,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192
```

Character Error Rates (CERs) listed below are produced by the checkpoint of the 18th epoch using BPE model ( # tokens is 2000, byte fallback enabled).

| Datasets | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech | WenetSpeech | WenetSpeech |
|--------------------------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|--------------------|-------------------------|---------------------|
|  Zipformer  CER   (%) |  eval | test | dev | test | dev | test | test | dev | test | dev phase1 | dev phase2 | test | dev | test meeting | test net |
| CTC Greedy Streaming |  26.50 | 28.10| 1.71 | 1.97| 3.89| 4.06 | 17.23 | 3.69 | 2.87 | 8.14 | 3.61 |9.51 | 6.11 | 8.13 | 10.62 |
| CTC Greedy Offline |  23.47 | 25.02 | 1.39 | 1.50 | 3.15 | 3.41 | 15.14 | 3.07 | 2.37 | 6.06 | 2.90 | 7.13 | 5.40 | 6.52 | 9.64 |
| Transducer Greedy Offline   |  23.16 | 24.78 | 1.33 | 1.38 | 3.06 | 3.23 | 15.36 | 2.54 | 2.09 | 5.24 | 2.28 | 6.26 | 4.87 | 6.26 | 7.07 |
| Transducer Greedy Streaming   |  26.83|28.74 | 1.75 | 1.91 | 3.84 | 4.12 | 17.83 | 3.23 | 2.71 | 7.31 | 3.16 | 8.69 | 5.71 | 7.91 | 8.54 |

Pre-trained model can be found here : https://huggingface.co/yuekai/icefall-asr-multi-zh-hans-zipformer-large

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
