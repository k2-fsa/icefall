# Introduction

Key features of VITS:

Combines VAE (Variational Autoencoder), normalizing flow, and GAN (adversarial training with a discriminator).
Uses Monotonic Alignment Search (MAS) — the model learns the alignment between text and audio automatically (no need for separate forced alignment like in older models).
Supports multi-speaker training (VCTK has ~109 different English speakers).
Generates natural-sounding speech with good prosody and voice quality.

The notebook uses the icefall implementation of VITS (generator + discriminator).

![alt text](image.png)

# Data Preparation

Run `prepare.sh` to download and prepare the data. All stages are run by default.

**Option A — Download automatically (default):**
```bash
bash prepare.sh
```

**Option B — Use pre-existing local data (skip download):**

If you already have the VCTK corpus available locally (e.g. from [Kaggle](https://www.kaggle.com/datasets/pratt3000/vctk-corpus)
or another source), pass `--local-data-dir` to skip Stage 0 download:

```bash
bash prepare.sh --local-data-dir /path/to/your/VCTK
```

This will create a symlink at `download/VCTK` pointing to your local copy,
so all subsequent stages work without any modification.

# VITS

This recipe provides a VITS model trained on the VCTK dataset.

Pretrained model can be found [here](https://huggingface.co/zrjin/icefall-tts-vctk-vits-2024-03-18), note that this model was pretrained on the Edinburgh DataShare VCTK dataset.

For tutorial and more details, please refer to the [VITS documentation](https://k2-fsa.github.io/icefall/recipes/TTS/vctk/vits.html).

The training command is given below:
```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./vits/train.py \
  --world-size 4 \
  --num-epochs 1000 \
  --start-epoch 1 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 350
```

To inference, use:
```
./vits/infer.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 500
```