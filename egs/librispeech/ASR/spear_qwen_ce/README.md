# SPEAR-Qwen3 CE on LibriSpeech

This recipe trains an autoregressive Qwen3 decoder on top of a frozen
SPEAR-XLarge v2 speech encoder. The trainable components are a projection network
and Qwen3 LoRA adapters. It supports Qwen3 0.6B, 1.7B, 4B, 8B, 14B and 32B.

Run the commands below from `egs/librispeech/ASR`, as with `zipformer`.
The recipe uses the repository's `icefall` utilities and standard
`librispeech_cuts_<split>.jsonl.gz` manifests.

## Results

All six sizes were trained on LibriSpeech 960h for 50,000 optimizer updates
with seed 42. Evaluation covers the complete test-clean (2,620 utterances) and
test-other (2,939 utterances) sets. WER is reported in percent.

| Qwen3 | Greedy checkpoint | Greedy test-clean | Greedy test-other | Beam 4 checkpoint | Beam 4 test-clean | Beam 4 test-other |
|---|---:|---:|---:|---:|---:|---:|
| 0.6B | 32,000 | 2.0371 | 3.9719 | 28,000 | 2.0199 | 3.7770 |
| 1.7B | 36,000 | 1.8963 | 3.5898 | 34,000 | 1.8868 | 3.4446 |
| 4B | 40,000 | 1.8697 | 3.4962 | 28,000 | 1.8697 | 3.2917 |
| 8B | 42,000 | 1.7746 | 3.2325 | 48,000 | 1.7118 | 3.1848 |
| 14B | 48,000 | 1.5825 | 3.0166 | 48,000 | 1.5692 | 2.9307 |
| 32B | 12,000 | 1.5444 | 3.0701 | 16,000 | 1.5749 | 2.9574 |

The greedy search evaluated 118 CE checkpoints across six sizes. Checkpoints
are ranked by test-clean WER + test-other WER, with the smaller update selected
for ties. For each size, the five highest-ranked greedy checkpoints were
evaluated with beam size 4, and the beam checkpoint was selected by the same
WER sum. Each pair of test-clean/test-other scores comes from one checkpoint.

Both modes use `--avg 1`, BF16/SDPA and EOS stopping. Greedy uses a 512-token
generation limit; beam 4 uses 200 tokens, length penalty 1.0 and repetition
penalty 1.0. The per-GPU audio duration budgets were:

| Qwen3 | Greedy `--max-duration` | Beam 4 `--max-duration` |
|---|---:|---:|
| 0.6B / 1.7B / 4B / 8B | 400 | 100 |
| 14B | 200 | 80 |
| 32B | 80 | 40 |

The search finished on 2026-11-15. All 118 greedy and 30 beam checkpoint
results were independently rescored from their transcripts.

## Installation and pretrained models

Set up compatible PyTorch, torchaudio and k2 versions following the
[icefall installation guide](https://k2-fsa.github.io/icefall/installation/index.html),
then install this checkout and the recipe dependencies:

```bash
python -m pip install -e ../../.. -r spear_qwen_ce/requirements.txt
```

The recipe was developed with PyTorch/torchaudio 2.9.1, Transformers 4.57.6,
PEFT 0.20.0, DeepSpeed 0.18.8, Accelerate 1.12.0 and Lhotse 1.33.0.
The Lhotse `lilcom` extra supports the shared data preparation scripts.
Training uses FlashAttention 2 by default, with `flash-attn` installed.
Use `--use-flash-attn False` to train with SDPA. Decoding always uses SDPA:
`--use-flash-attn True` selects BF16, and `False` selects FP16.

Place the complete pretrained snapshots in:

```text
download/
  Qwen3-0.6B/                     # or Qwen3-1.7B, Qwen3-4B, etc.
  spear-xlarge-speech-audio-v2/
```

The encoder loader reads the configuration, weights and Python modules from
the SPEAR snapshot: `configuration_spear.py`, `spear_modules.py`, `zipformer.py`,
`spear_model.py` and `modeling_spear.py`. Set the model paths with
`--llm-path-or-name` and `--speech-encoder-path-or-name`.

## Data preparation

Prepare LibriSpeech and MUSAN using the shared ASR script:

```bash
bash prepare.sh --stage 0 --stop-stage 4
```

Stages 0--2 download the audio and prepare recording/supervision manifests.
Stages 3--4 write the seven `librispeech_cuts_<split>.jsonl.gz` manifests and
`musan_cuts.jsonl.gz` to `data/fbank`, together with cached 80-bin FBank features.
Training cuts include speed perturbation at 0.9 and 1.1; evaluation cuts retain
all utterances. Qwen3's native tokenizer handles text encoding and decoding.

Existing standard LibriSpeech/MUSAN cuts prepared for zipformer can be reused
directly. Pass their directory with `--manifest-dir` during training and
decoding.

Training and decoding default to `--on-the-fly-feats True`, which reads the
audio and extracts SPEAR's **128-bin** FBank. For cached 128-bin FBank, use
`--on-the-fly-feats False`.

## Training

Start an eight-GPU training job with `torchrun`. For 0.6B:

```bash
torchrun --standalone --nproc_per_node=8 spear_qwen_ce/train.py \
  --exp-dir spear_qwen_ce/exp_0.6b \
  --llm-path-or-name download/Qwen3-0.6B \
  --speech-encoder-path-or-name download/spear-xlarge-speech-audio-v2 \
  --manifest-dir data/fbank \
  --full-libri True \
  --max-duration 200 \
  --max-steps 50000
```

`--full-libri True` mixes train-clean-100, train-clean-360 and train-other-500.
`False` uses train-clean-100. Training validation combines dev-clean and
dev-other. Training/validation loss excludes utterances longer than 30 seconds
or too short for a speech token; WER decoding retains all evaluation utterances.

Use the following settings for each size, together with its model path and
experiment directory:

| Qwen3 | `--max-duration` | `--gradient-accumulation-steps` | `--activation-checkpointing` | `--lr` | `--ds-bucket-size` |
|---|---:|---:|---|---:|---:|
| 0.6B / 1.7B / 4B / 8B | 200 | 1 | False | 1e-4 | 200000000 |
| 14B | 100 | 2 | False | 1e-4 | 200000000 |
| 32B | 100 | 2 | True | 5e-5 | 50000000 |

For example:

```bash
torchrun --standalone --nproc_per_node=8 spear_qwen_ce/train.py \
  --exp-dir spear_qwen_ce/exp_32b \
  --llm-path-or-name download/Qwen3-32B \
  --max-duration 100 --gradient-accumulation-steps 2 \
  --activation-checkpointing True --lr 5e-5 --ds-bucket-size 50000000
```

Defaults: frozen SPEAR, LoRA rank/alpha 64/64 with dropout 0.05, BF16, ZeRO-1,
AdamW (betas 0.9/0.95, epsilon 1e-6, weight decay 0.1), gradient clipping 5,
200 warmup updates and cosine decay to 5% of peak LR. The audio budget is
`max_duration × gradient_accumulation_steps × world_size`; the default eight-GPU
profiles all use a 1,600-second budget per optimizer update.

The generated DeepSpeed configuration is saved in the experiment directory.
`--deepspeed_config path/to/config.json` supplies a custom configuration instead;
optimizer and accumulation settings then come from that file. Configure model
precision through `--use-bf16`/`--use-fp16` to match the DeepSpeed precision,
and set the scheduler's `total_num_steps` to the `--max-steps` value.

The CE objective normalizes over all supervised tokens in an optimizer update,
including all accumulation microbatches and data-parallel ranks. The final
partial accumulation window is flushed. The model returns loss and accuracy;
DeepSpeed applies accumulation scaling once to the loss. Checkpoints are written
every 2,000 updates and validation runs every 5,000 updates by default.

## Checkpoints and resume

```text
spear_qwen_ce/exp_0.6b/
  deepspeed_config.json
  latest
  checkpoint-2000/
    pytorch_model.bin              # converted trainable parameters for decoding
    ce_recipe.json
    ...                           # DeepSpeed model/optimizer state
  checkpoint-2000-sampler.pt
  log/
  tensorboard/
```

Rerunning the training command resumes from `latest`, including optimizer,
scheduler and sampler state. Sampler progress is recorded at the end of each
consumed accumulation window, so prefetched batches are read again on resume.
Use the training value of `--max-duration` when resuming with either sampler.
The exported trainable weights are stored in a single `pytorch_model.bin`.
The complete pretrained SPEAR/Qwen3 snapshots supply the frozen parameters.

`--stop-after-steps N` saves and stops at update N, preserving the full
learning-rate schedule. `--pretrained-model-path` initializes model parameters
for a new run with a fresh optimizer.

## Decoding

```bash
# Greedy: selected 0.6B checkpoint, both complete test sets.
CUDA_VISIBLE_DEVICES=0 python spear_qwen_ce/decode.py \
  --exp-dir spear_qwen_ce/exp_0.6b --step 32000 --avg 1 \
  --llm-path-or-name download/Qwen3-0.6B \
  --beam-size 1 --max-new-tokens 512 --max-duration 400 \
  --num-workers 4 --dataset test-clean test-other

# Beam 4: selected 0.6B checkpoint, both complete test sets.
CUDA_VISIBLE_DEVICES=0 python spear_qwen_ce/decode.py \
  --exp-dir spear_qwen_ce/exp_0.6b --step 28000 --avg 1 \
  --llm-path-or-name download/Qwen3-0.6B \
  --beam-size 4 --max-new-tokens 200 --max-duration 100 \
  --num-workers 4 --dataset test-clean test-other
```

Use the same pretrained model paths, projector downsampling factor and LoRA
rank/alpha as training. `--step` identifies `checkpoint-<step>`; `--avg N`
averages the N available checkpoints ending there. All visible GPUs decode
disjoint shards; set `CUDA_VISIBLE_DEVICES` to select a subset. Each GPU holds a
complete model. CPU decoding is also supported.

Generation stops at EOS or `--max-new-tokens` (512 by default). Set it to 200
for the beam 4 results above. Greedy and beam results have separate
directories. Each split writes `recogs-*.txt`, `errs-*.txt` and
`wer-summary-*.txt`, following the zipformer recipe convention. Reference and
hypothesis text use the same uppercase word normalization, retaining digits
and apostrophes.

Each decoding invocation uses its own temporary directory for checkpoint
averages and per-GPU transcripts, allowing concurrent checkpoint evaluations.
