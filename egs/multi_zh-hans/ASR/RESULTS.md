## Results

### Multi Chinese datasets (without datatang 200h) finetuning results on Whisper-large-v2
#### Whisper
[./whisper](./whisper)

Character Error Rates (CERs) listed below are produced by the checkpoint of the second epoch using greedy search.

| Datasets | alimeeting | alimeeting | aishell-1 | aishell-1 | aishell-2 | aishell-2 | aishell-4 | magicdata | magicdata | kespeech-asr | kespeech-asr | kespeech-asr | WenetSpeech  |
|--------------------------------|-------------------|--------------|----------------|-------------|------------------|-------------|------------------|------------------|-------------|-----------------------|-----------------------|-------------|-------------------|
|  Split |           eval| test | dev | test | dev| test | test      | dev| test | dev phase1 | dev phase2 | test | test meeting |
| Greedy Search |  23.22 | 28.24 | 0.61 | 0.66 | 2.67 | 2.80 | 16.61 | 2.56 | 2.21 | 4.73 | 1.90 | 5.98 |                    8.13 |

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
