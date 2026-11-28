# SPEAR-Qwen3 CE on LibriSpeech

SPEAR-Qwen3 CE uses a frozen SPEAR-XLarge v2 encoder, a trainable projector,
and an LLM with LoRA rank/alpha 64/64.

Run recipe commands from `egs/librispeech/ASR`.

## Results

All models were trained for 50,000 steps on LibriSpeech 960h with seed 42,
globally shuffled cuts, and speed factors 0.9/1.0/1.1.

| Qwen3 | Greedy search step | test-clean | test-other | Beam search step | test-clean | test-other |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B | 24,000 | 2.0371 | 3.8477 | 18,000 | 2.0047 | 3.6509 |
| 1.7B | 32,000 | 1.9648 | 3.7312 | 18,000 | 1.8602 | 3.6051 |
| 4B | 44,000 | 1.8526 | 3.4026 | 44,000 | 1.8202 | 3.2707 |
| 8B | 48,000 | 1.7460 | 3.2784 | 48,000 | 1.7232 | 3.1752 |
| 14B | 46,000 | 1.5596 | 3.0759 | 46,000 | 1.5368 | 2.9956 |
| 32B | 34,000 | 1.6833 | 2.9823 | 34,000 | 1.6547 | 2.9020 |

WER (%), `avg=1`, beam size 4.

## Installation

Install icefall, PyTorch, torchaudio, and k2 following the
[icefall installation guide](https://k2-fsa.github.io/icefall/installation/index.html),
then install the training and inference dependencies:

```bash
python3 -m pip install -r spear_qwen_ce/requirements.txt
```

## Pretrained models

[Pretrained models](https://huggingface.co/yfyeung/icefall-asr-librispeech-spear-qwen3-ce).
`download.py` downloads the checkpoints, SPEAR encoder, and Qwen3 base model.

```bash
export HF_ENDPOINT="https://hf-mirror.com"
python3 spear_qwen_ce/download.py --size 0.6B --output-dir download/spear-qwen3-ce

CUDA_VISIBLE_DEVICES=0 python3 spear_qwen_ce/pretrained.py \
  --checkpoint download/spear-qwen3-ce/exp/0.6B/checkpoint-24000/pytorch_model.bin \
  --llm-path-or-name download/spear-qwen3-ce/base_models/Qwen3-0.6B \
  --speech-encoder-path-or-name download/spear-qwen3-ce/encoder \
  --decoding-method greedy_search \
  /path/to/audio.wav
```

For beam search, use the corresponding step from the table and
`--decoding-method beam_search --beam-size 4`.

## Data preparation

```bash
bash prepare.sh --stage 0 --stop-stage 0
python3 spear_qwen_ce/prepare_manifests.py \
  --audio-root download/LibriSpeech --musan-root download/musan \
  --output-dir data/spear_fbank --num-jobs 15
```

## Training

The following command trains Qwen3 0.6B on eight GPUs:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 spear_qwen_ce/train.py \
  --deepspeed --exp-dir spear_qwen_ce/exp_0.6b \
  --llm-path-or-name download/spear-qwen3-ce/base_models/Qwen3-0.6B \
  --speech-encoder-path-or-name download/spear-qwen3-ce/encoder \
  --manifest-dir data/spear_fbank --train-manifest librispeech_cuts_train-all-shuf.jsonl.gz \
  --full-libri True --on-the-fly-feats True --max-duration 200 \
  --gradient-accumulation-steps 1 --activation-checkpointing False \
  --lr 1e-4 --max-steps 50000 --seed 42
```

Use the following settings for each Qwen3 size:

| Qwen3 | `--max-duration` | `--gradient-accumulation-steps` | `--activation-checkpointing` | `--lr` | `--ds-bucket-size` |
|---|---:|---:|---|---:|---:|
| 0.6B / 1.7B / 4B / 8B | 200 | 1 | False | 1e-4 | 200000000 |
| 14B | 100 | 2 | False | 1e-4 | 200000000 |
| 32B | 100 | 2 | True | 5e-5 | 50000000 |

Training uses BF16, ZeRO-1, MUSAN, SpecAugment, and online
128-bin FBank features. The learning rate uses 200 warmup steps and cosine decay.
Rerun the command to resume from `latest`.

## Decoding

Decode test-clean and test-other:

```bash
CUDA_VISIBLE_DEVICES=0 python3 spear_qwen_ce/decode.py \
  --exp-dir download/spear-qwen3-ce/exp/0.6B --step 24000 --avg 1 \
  --llm-path-or-name download/spear-qwen3-ce/base_models/Qwen3-0.6B \
  --speech-encoder-path-or-name download/spear-qwen3-ce/encoder \
  --manifest-dir data/spear_fbank --on-the-fly-feats True \
  --decoding-method greedy_search --max-duration 800 \
  --max-new-tokens 512 --num-workers 4 --dataset test-clean test-other

CUDA_VISIBLE_DEVICES=0 python3 spear_qwen_ce/decode.py \
  --exp-dir download/spear-qwen3-ce/exp/0.6B --step 18000 --avg 1 \
  --llm-path-or-name download/spear-qwen3-ce/base_models/Qwen3-0.6B \
  --speech-encoder-path-or-name download/spear-qwen3-ce/encoder \
  --manifest-dir data/spear_fbank --on-the-fly-feats True \
  --decoding-method beam_search --beam-size 4 --max-duration 400 \
  --max-new-tokens 512 --num-workers 4 --dataset test-clean test-other
```

Decoding `--max-duration` per GPU:

| Qwen3 | Greedy search | Beam search, beam size 4 |
|---|---:|---:|
| 0.6B / 1.7B / 4B / 8B | 800 | 400 |
| 14B | 800 | 200 |
| 32B | 800 | 60 |
