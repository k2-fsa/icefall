## ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching


[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](http://arxiv.org/abs/2506.13053)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://zipvoice.github.io/)


## Overview
ZipVoice is a high-quality zero-shot TTS model with a small model size and fast inference speed.
#### Key features:

- Small and fast: only 123M parameters.

- High-quality: state-of-the-art voice cloning performance in speaker similarity, intelligibility, and naturalness.

- Multi-lingual: support Chinese and English.


## News
**2025/06/16**: 🔥 ZipVoice is released.


## Installation

* Clone icefall repository and change to zipvoice directory:

```bash
git clone https://github.com/k2-fsa/icefall.git
cd icefall/egs/zipvoice
```

* Create a Python virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

* Install the required packages:

```bash
# Install pytorch and k2.
# If you want to use different versions, please refer to https://k2-fsa.org/get-started/k2/ for details.
# For users in China mainland, please refer to https://k2-fsa.org/zh-CN/get-started/k2/

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install k2==1.24.4.dev20250208+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html

# Install other dependencies.
pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
pip install -r requirements.txt
```

## Usage

To generate speech with our pre-trained ZipVoice or ZipVoice-Distill models, use the following commands (Required models will be downloaded from HuggingFace):

### 1. Inference of a single sentence:
```bash
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice_distill" \
    --prompt-wav prompt.wav \
    --prompt-text "I am the transcription of the prompt wav." \
    --text "I am the text to be synthesized." \
    --res-wav-path result.wav

# Example with a pre-defined prompt wav and text
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice_distill" \
    --prompt-wav assets/prompt-en.wav \
    --prompt-text "Some call me nature, others call me mother nature. I've been here for over four point five billion years, twenty two thousand five hundred times longer than you." \
    --text "Welcome to use our tts model, have fun!" \
    --res-wav-path result.wav
```

### 2. Inference of a list of sentences:
```bash
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice_distill" \
    --test-list test.tsv \
    --res-dir results/test
```

- `--model-name` can be `zipvoice` or `zipvoice_distill`, which are models before and after distillation, respectively.
- Each line of `test.tsv` is in the format of `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.


> **Note:**  If you having trouble connecting to HuggingFace, try:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Training Your Own Model

The following steps show how to train a model from scratch on Emilia and LibriTTS datasets, respectively.

### 0. Install dependencies for training

```bash
pip install -r ../../requirements.txt
```

### 1. Data Preparation

#### 1.1. Prepare the Emilia dataset

```bash
bash scripts/prepare_emilia.sh
```

See [scripts/prepare_emilia.sh](scripts/prepare_emilia.sh) for step by step instructions.

#### 1.2 Prepare the LibriTTS dataset

```bash
bash scripts/prepare_libritts.sh
```

See [scripts/prepare_libritts.sh](scripts/prepare_libritts.sh) for step by step instructions.

### 2. Training

#### 2.1 Traininig on Emilia

<details>
<summary>Expand to view training steps</summary>

##### 2.1.1 Train the ZipVoice model

- Training:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_flow.py \
        --world-size 8 \
        --use-fp16 1 \
        --dataset emilia \
        --max-duration 500 \
        --lr-hours 30000 \
        --lr-batches 7500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --num-epochs 11 \
        --exp-dir zipvoice/exp_zipvoice
```

-  Average the checkpoints to produce the final model:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/generate_averaged_model.py \
      --epoch 11 \
      --avg 4 \
      --distill 0 \
      --token-file data/tokens_emilia.txt \
      --dataset "emilia" \
      --exp-dir ./zipvoice/exp_zipvoice
# The generated model is zipvoice/exp_zipvoice/epoch-11-avg-4.pt
```

##### 2.1.2. Train the ZipVoice-Distill model (Optional)

- The first-stage distillation:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0005 \
        --max-duration 500 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
        --num-updates 60000 \
        --distill-stage "first" \
        --exp-dir zipvoice/exp_zipvoice_distill_1stage
```

- Average checkpoints for the second-stage initialization:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/generate_averaged_model.py \
      --iter 60000 \
      --avg 7 \
      --distill 1 \
      --token-file data/tokens_emilia.txt \
      --dataset "emilia" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_1stage
# The generated model is zipvoice/exp_zipvoice_distill_1stage/iter-60000-avg-7.pt
```

-  The second-stage distillation:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "emilia" \
        --base-lr 0.0001 \
        --max-duration 200 \
        --token-file "data/tokens_emilia.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_distill_1stage/iter-60000-avg-7.pt \
        --num-updates 2000 \
        --distill-stage "second" \
        --exp-dir zipvoice/exp_zipvoice_distill_new
```
</details>


#### 2.2 Traininig on LibriTTS

<details>
<summary>Expand to view training steps</summary>

##### 2.2.1 Train the ZipVoice model

- Training:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_flow.py \
        --world-size 8 \
        --use-fp16 1 \
        --dataset libritts \
        --max-duration 250 \
        --lr-epochs 10 \
        --lr-batches 7500 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --num-epochs 60 \
        --exp-dir zipvoice/exp_zipvoice_libritts
```

- Average the checkpoints to produce the final model:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/generate_averaged_model.py \
      --epoch 60 \
      --avg 10 \
      --distill 0 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_libritts
# The generated model is zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt
```

##### 2.1.2 Train the ZipVoice-Distill model (Optional)

- The first-stage distillation:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "libritts" \
        --base-lr 0.001 \
        --max-duration 250 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt \
        --num-epochs 6 \
        --distill-stage "first" \
        --exp-dir zipvoice/exp_zipvoice_distill_1stage_libritts
```

- Average checkpoints for the second-stage initialization:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 ./zipvoice/generate_averaged_model.py \
      --epoch 6 \
      --avg 3 \
      --distill 1 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_1stage_libritts
# The generated model is zipvoice/exp_zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt
```

- The second-stage distillation:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/train_distill.py \
        --world-size 8 \
        --use-fp16 1 \
        --tensorboard 1 \
        --dataset "libritts" \
        --base-lr 0.001 \
        --max-duration 250 \
        --token-file "data/tokens_libritts.txt" \
        --manifest-dir "data/fbank" \
        --teacher-model zipvoice/exp_zipvoice_distill_1stage_libritts/epoch-6-avg-3.pt \
        --num-epochs 6 \
        --distill-stage "second" \
        --exp-dir zipvoice/exp_zipvoice_distill_libritts
```

- Average checkpoints to produce the final model:

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 ./zipvoice/generate_averaged_model.py \
      --epoch 6 \
      --avg 3 \
      --distill 1 \
      --token-file data/tokens_libritts.txt \
      --dataset "libritts" \
      --exp-dir ./zipvoice/exp_zipvoice_distill_libritts
# The generated model is ./zipvoice/exp_zipvoice_distill_libritts/epoch-6-avg-3.pt
```
</details>


### 3. Inference with the trained model

#### 3.1 Inference with the model trained on Emilia
<details>
<summary>Expand to view inference commands.</summary>

##### 3.1.1 ZipVoice model before distill:
```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
      --distill 0 \
      --token-file "data/tokens_emilia.txt" \
      --test-list test.tsv \
      --res-dir results/test \
      --num-step 16 \
      --guidance-scale 1
```

##### 3.1.2 ZipVoice-Distill model before distill:
```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_distill/checkpoint-2000.pt \
      --distill 1 \
      --token-file "data/tokens_emilia.txt" \
      --test-list test.tsv \
      --res-dir results/test_distill \
      --num-step 8 \
      --guidance-scale 3
```
</details>


#### 3.2 Inference with the model trained on LibriTTS

<details>
<summary>Expand to view inference commands.</summary>

##### 3.2.1 ZipVoice model before distill:
```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_libritts/epoch-60-avg-10.pt \
      --distill 0 \
      --token-file "data/tokens_libritts.txt" \
      --test-list test.tsv \
      --res-dir results/test_libritts \
      --num-step 8 \
      --guidance-scale 1 \
      --target-rms 1.0 \
      --t-shift 0.7
```

##### 3.2.2 ZipVoice-Distill model before distill

```bash
export PYTHONPATH=../../:$PYTHONPATH
python3 zipvoice/infer.py \
      --checkpoint zipvoice/exp_zipvoice_distill/epoch-6-avg-3.pt \
      --distill 1 \
      --token-file "data/tokens_libritts.txt" \
      --test-list test.tsv \
      --res-dir results/test_distill_libritts \
      --num-step 4 \
      --guidance-scale 3 \
      --target-rms 1.0 \
      --t-shift 0.7
```
</details>

### 4. Evaluation on benchmarks

See [local/evaluate.sh](local/evaluate.sh) for details of objective metrics evaluation
on three test sets, i.e., LibriSpeech-PC test-clean, Seed-TTS test-en and Seed-TTS test-zh.


## Citation

```bibtex
@article{zhu-2025-zipvoice,
      title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching}, 
      author={Han Zhu and Wei Kang and Zengwei Yao and Liyong Guo and Fangjun Kuang and Zhaoqing Li and Weiji Zhuang and Long Lin and Daniel Povey}
      journal={arXiv preprint arXiv:2506.13053},
      year={2025},
}
```
