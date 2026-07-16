# Train VITS Model From Scratch

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
!bash prepare.sh --stage 1 --stop_stage 6
```

### Cell 3: Train Model
```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

!CUDA_VISIBLE_DEVICES="0" python vits/train.py \
  --world-size 1 \
  --num-epochs 1000 \
  --start-epoch 1 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt \
  --max-duration 350
```

### Cell 4: View TensorBoard Logs
```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/icefall/egs/vctk/TTS/vits/exp/tensorboard
```

### Cell 5: Export to ONNX (After Training)
```bash
%cd /kaggle/working/icefall/egs/vctk/TTS

!python vits/export-onnx.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt
```
