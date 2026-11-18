# Visual Speech Recognition - Grid Corpus - AV-HuBERT features

Lipreading on GRID with a frozen AV-HuBERT Base front-end (layer 8, mean-face
mouth ROIs from `local/meanface_roi.py`) and a 6-layer Conformer-CTC head
(`conformer_ctc2`, no subsampling, 25 fps), decoded through a phone HLG. Test
speakers s1, s2, s20 and s22 are unseen in training.

| System | WER (unseen speakers, 3 seeds) |
|---|---|
| `conformer_ctc2`, frozen front-end | 7.15% |
| `conformer_ctc2`, fine-tuned front-end (`train_ft.py`) | 6.25% |

[RESULTS.md](RESULTS.md) has the commands, the overlapped-speaker protocol and
pretrained models.

### Installation
The recipe needs Python 3.8 (AV-HuBERT and its fairseq fork do not support newer versions) and a CUDA 12.1 GPU: `install.sh` installs a k2 wheel built for CUDA 12.1, PyTorch 2.4.1 and Python 3.8.

1. Create and activate a conda environment:
   ```
   conda create -n icefall-vsr python=3.8 -y
   conda activate icefall-vsr
   ```
2. From icefall's root, run the installer:
   ```
   cd egs/grid/VSR
   ./install.sh
   ```
   It runs three stages (resume from one with `--stage N`):
   1. k2, lhotse and icefall's own requirements;
   2. AV-HuBERT, cloned into `av_hubert/`, and its fairseq fork;
   3. this recipe's requirements (`grid-requirements.txt`) and dlib (from conda-forge).
3. Put icefall on your Python path, as the installer prints at the end (add it to `~/.bashrc` or the environment's activate hook to keep it):
   ```
   export PYTHONPATH=/path/to/icefall:$PYTHONPATH
   ```
4. Prepare the data and features:
   ```
   ./prepare.sh
   ```
   This downloads GRID, the dlib landmark model, the mean-face reference and the AV-HuBERT Base checkpoint, then builds manifests, layer-8 features, the phone lexicon and the decoding graphs (stages 0 to 9; `--stage` and `--stop-stage` select a range).


### Recognising a single clip
`conformer_ctc2/lipreader.py` runs the whole pipeline (landmarks, mean-face mouth crop, AV-HuBERT, Conformer, HLG) on one video and prints the recognised sentence:
```
./conformer_ctc2/lipreader.py --checkpoint conformer_ctc2/exp/pretrained.pt \
  --roi-mode meanface --layer 8 --tokens data/lang_phone/tokens_model.txt \
  --method 1best --HLG data/lang_phone/HLG.pt --words-file data/lang_phone/words.txt \
  --avhubert-ckpt download/avhubert-ckpts/base_vox_iter5.pt download/grid-corpus/s33/pgwh8n.mpg
```
