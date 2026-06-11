# Inference with Pretrained VITS Model

### Cell 1: Install Dependencies
```bash
# Install icefall repo and requirements
!git clone https://github.com/k2-fsa/icefall.git /kaggle/working/icefall
!pip install -r /kaggle/working/icefall/requirements.txt
!grep -v 'numba' /kaggle/working/icefall/requirements-tts.txt | pip install -r /dev/stdin
!pip install "numba>=0.59.0"

# Install lhotse (audio dataset toolkit)
!pip install lhotse

# Install k2 (must match CUDA 12.8 + PyTorch 2.10.0)
!pip install k2==1.24.4.dev20260306+cuda12.8.torch2.10.0 -f https://k2-fsa.github.io/k2/cuda.html

# Install piper_phonemize and register icefall
!pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
!pip install -e /kaggle/working/icefall
```

### Cell 2: Prepare Dataset
```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

# Symlink VCTK data to bypass download stage
!mkdir -p download
!ln -sfv /kaggle/input/datasets/ download/VCTK

# Build monotonic_align C extension
!bash prepare.sh --stage -1 --stop_stage -1

# Create manifests, spectrograms, tokens, and data splits
!bash prepare.sh --stage 1 --stop_stage 4
```

### Cell 3: Download Pretrained Model
```python
from huggingface_hub import hf_hub_download
import os, shutil

MODEL_ID = "zrjin/icefall-tts-vctk-vits-2024-03-18"
BASE_DIR  = "/kaggle/working/icefall/egs/vctk/TTS"

os.makedirs(f"{BASE_DIR}/vits/exp", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data", exist_ok=True)

# Download checkpoint and move to correct directory
hf_hub_download(repo_id=MODEL_ID, filename="exp/epoch-1000.pt", local_dir=BASE_DIR)
shutil.copy2(f"{BASE_DIR}/exp/epoch-1000.pt", f"{BASE_DIR}/vits/exp/epoch-1000.pt")

# Download tokens and speakers
hf_hub_download(repo_id=MODEL_ID, filename="data/tokens.txt", local_dir=BASE_DIR)
hf_hub_download(repo_id=MODEL_ID, filename="data/speakers.txt", local_dir=BASE_DIR)

print("Pretrained model downloaded and moved to correct directories.")
```

### Cell 4: Run Inference
```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

!CUDA_VISIBLE_DEVICES="0" python vits/infer.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 500
```

### Cell 5: Play Generated Audio
```python
import os
from IPython.display import Audio, display

wav_dir = "/kaggle/working/icefall/egs/vctk/TTS/vits/exp/infer/epoch-1000/wav"
# Choose to play audio from test set directory
wav_dir_test = os.path.join(wav_dir, "test")
wav_files = sorted(os.listdir(wav_dir_test))

# Play the first 3 generated audio files
for f in wav_files[:3]:
    print(f)
    display(Audio(os.path.join(wav_dir_test, f)))
```
