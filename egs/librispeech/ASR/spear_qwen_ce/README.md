# SPEAR-Qwen3 CE on LibriSpeech

This recipe trains an autoregressive Qwen3 decoder on top of a frozen
SPEAR-XLarge v2 speech encoder. The trainable components are a projection network
and Qwen3 LoRA adapters. It supports Qwen3 0.6B, 1.7B, 4B, 8B, 14B and 32B.

Run the commands below from `egs/librispeech/ASR`, as with `zipformer`.
The recipe uses the repository's `icefall` utilities and standard
`librispeech_cuts_<split>.jsonl.gz` manifests. The CE model, token-normalized
loss, checkpoint format and resume behavior are retained from the LibriSpeech
`spear_qwen_e5/ce` implementation.

## Installation and pretrained models

Install the repository with a compatible PyTorch, torchaudio and k2 environment,
then install the recipe-specific dependencies:

```bash
python -m pip install -r spear_qwen_ce/requirements.txt
```

The recipe was developed with PyTorch 2.9.1, Transformers 4.57.6, PEFT 0.20.0,
DeepSpeed 0.18.8 and Lhotse 1.33.0. Training uses FlashAttention 2 by default;
install a compatible `flash-attn` build, or pass `--use-flash-attn False` to
train with SDPA. Decoding uses BF16/SDPA with `--use-flash-attn True`.

Place the complete pretrained snapshots in:

```text
download/
  Qwen3-0.6B/                     # or Qwen3-1.7B, Qwen3-4B, etc.
  spear-xlarge-speech-audio-v2/
```

The SPEAR snapshot must include its configuration, weights and all five Python
modules: `configuration_spear.py`, `spear_modules.py`, `zipformer.py`,
`spear_model.py` and `modeling_spear.py`. The encoder loader imports these modules
from the supplied snapshot. Override the two model paths with
`--llm-path-or-name` and `--speech-encoder-path-or-name` as needed.

## Data preparation

Prepare the LibriSpeech and MUSAN recording/supervision manifests using the
existing ASR script, then create cuts for this recipe:

```bash
bash prepare.sh --stage 0 --stop-stage 2
python spear_qwen_ce/prepare.py
```

`spear_qwen_ce/prepare.py` writes the standard seven LibriSpeech cut manifests
and `musan_cuts.jsonl.gz` to `data/fbank`. It applies speed perturbation at 0.9
and 1.1 to training data only. It leaves existing cut files untouched. To change
the preparation settings, use a new `--output-dir` and pass that directory as
`--manifest-dir` during training and decoding.

You can also use existing standard LibriSpeech cuts prepared for zipformer.
Training and decoding default to `--on-the-fly-feats True`: SPEAR requires
**128-bin** FBank, whereas zipformer's cached features normally have 80 bins.
Disable online extraction only when the cached features have 128 bins.
Audio sources referenced by the manifests must remain accessible.

No BPE lexicon or decoding graph is required; this recipe uses Qwen3's tokenizer.

## Training

Each model size has a launcher with its duration, accumulation, learning rate
and activation checkpointing defaults:

```bash
bash spear_qwen_ce/run_0.6b_ce.sh
bash spear_qwen_ce/run_1.7b_ce.sh
bash spear_qwen_ce/run_4b_ce.sh
bash spear_qwen_ce/run_8b_ce.sh
bash spear_qwen_ce/run_14b_ce.sh
bash spear_qwen_ce/run_32b_ce.sh
```

Run one command per training job. Each starts eight GPUs by default and writes
to `spear_qwen_ce/exp_<size>`. Preview or override the defaults with:

```bash
bash spear_qwen_ce/run_32b_ce.sh --dry-run
bash spear_qwen_ce/run_4b_ce.sh --nproc-per-node 2 --use-flash-attn False \
  --exp-dir spear_qwen_ce/exp_4b_run2 --max-duration 100
```

`--python`, `--nproc-per-node` and `--master-port` configure the launcher; all
other options go to `train.py` and override the size profile. `PYTHON`,
`NPROC_PER_NODE`, `MODEL_ROOT` and the training overrides listed by `--help`
can also be set through the environment. Paths resolve from
`egs/librispeech/ASR`, including when a launcher is invoked by absolute path.
The launcher adds this checkout to `PYTHONPATH` and uses a free rendezvous port
unless `--master-port` or `MASTER_PORT` is set.

The equivalent direct training command for 0.6B is:

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

The following settings reproduce the model-size profiles of the original CE
recipe. Change the model path and experiment directory for each size:

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
optimizer, precision and accumulation settings then come from that file.
Keep its scheduler length consistent with `--max-steps`.

The CE objective normalizes over all supervised tokens in an optimizer update,
including all accumulation microbatches and data-parallel ranks. The final
partial accumulation window is flushed. Checkpoints are written every 2,000
updates and validation runs every 5,000 updates by default.

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
scheduler and sampler state. The complete pretrained SPEAR/Qwen3 snapshots are
still needed: checkpoints omit frozen parameters. Keep the model/data settings
and DeepSpeed configuration unchanged when resuming.

`--stop-after-steps N` saves and stops at update N without changing the full
learning-rate schedule. This is useful for a short training/resume check:

```bash
torchrun --standalone --nproc_per_node=1 spear_qwen_ce/train.py \
  --exp-dir spear_qwen_ce/exp_smoke \
  --max-duration 20 --num-workers 0 \
  --limit-train-cuts 128 --limit-valid-cuts 16 \
  --use-flash-attn False --stop-after-steps 2
```

Use a separate experiment directory for this check. `--resume False` refuses to
overwrite an existing completed checkpoint. `--pretrained-model-path` initializes
model parameters for a new run; it does not restore optimizer state.

## Decoding

```bash
# Greedy: both complete test sets.
python spear_qwen_ce/decode.py \
  --exp-dir spear_qwen_ce/exp_0.6b --step 50000 --avg 1 \
  --llm-path-or-name download/Qwen3-0.6B \
  --beam-size 1 --dataset test-clean test-other

# Beam search on dev or test splits.
python spear_qwen_ce/decode.py \
  --exp-dir spear_qwen_ce/exp_0.6b --step 50000 --avg 1 \
  --llm-path-or-name download/Qwen3-0.6B \
  --beam-size 4 --dataset dev-clean dev-other test-clean test-other
```

Use the same pretrained model paths, projector downsampling factor and LoRA
rank/alpha as training. `--step` identifies `checkpoint-<step>`; `--avg N`
averages the N available checkpoints ending there. All visible GPUs decode
disjoint shards; set `CUDA_VISIBLE_DEVICES` to select a subset. Each GPU holds a
complete model. CPU decoding is also supported.

Generation stops at EOS or `--max-new-tokens` (512 by default, matching the
original LibriSpeech CE evaluation). Greedy and beam results have separate
directories. Each split writes `recogs-*.txt`, `errs-*.txt` and
`wer-summary-*.txt`, following the zipformer recipe convention. Reference and
hypothesis text use the same uppercase word normalization, retaining digits
and apostrophes.

## Files

`train.py`, `decode.py`, `model.py`, `encoder.py` and `asr_datamodule.py` are the
main entry points and components. `resume_ce.py` preserves the CE checkpoint
format. `prepare.py` prepares cuts, and `scoring.py` defines word normalization.
The recipe is self-contained within this repository; it does not import the
original experimental checkout or require its launch scripts.

The six `run_<size>_ce.sh` entry points share `run_train.sh`.
Measured WERs, checkpoint selection and evaluation settings are documented in
[the ASR results](../RESULTS.md#spear-qwen3-ce).
