## Results

### [conformer_ctc2](./conformer_ctc2)

### 2026-10-03: phone lexicon, mean-face ROIs, AV-HuBERT layer 8

|  | WER (unseen speaker, GRID) |
|------------------------|-----|
| 1best, seed 42 | 7.14% |
| 1best, mean of 3 seeds (42/43/44) | 7.15% |

Features (`prepare.sh` stage 3, `--roi-mode meanface --layer 8`) go to `data/avhubert_mf_l8`. The training command is:
```bash
./conformer_ctc2/train.py \
  --exp-dir conformer_ctc2/exp \
  --manifest-dir data/avhubert_mf_l8 \
  --lang-dir data/lang_phone \
  --num-epochs 8 \
  --dropout 0.4 \
  --max-duration 384 \
  --on-the-fly-feats false \
  --enable-aug false
```

The decoding command is:
```bash
./conformer_ctc2/decode.py \
  --exp-dir conformer_ctc2/exp \
  --manifest-dir data/avhubert_mf_l8 \
  --lang-dir data/lang_phone \
  --method 1best \
  --epoch 8 \
  --avg 4 \
  --max-duration 384 \
  --on-the-fly-feats false \
  --enable-aug false
```

To export the averaged model (epochs 5 to 8) as `conformer_ctc2/exp/pretrained.pt`:
```bash
./conformer_ctc2/export.py --exp-dir conformer_ctc2/exp --tokens data/lang_phone/tokens.txt \
  --epoch 8 --avg 4 --jit false
```

You can find the pretrained model (seed 42, epochs 5 to 8 averaged) by visiting
<https://huggingface.co/ialmajai/icefall-vsr-grid-conformer-ctc2-phone-l8-2026-10-03>

```bash
# copy exp/pretrained.pt from the Hugging Face repo above to $PWD/epoch-9999.pt

./conformer_ctc2/decode.py \
  --exp-dir $PWD \
  --manifest-dir data/avhubert_mf_l8 \
  --lang-dir data/lang_phone \
  --method 1best \
  --epoch 9999 \
  --avg 1 \
  --use-averaged-model false \
  --max-duration 384 \
  --on-the-fly-feats false \
  --enable-aug false
```

#### Overlapped speakers

The LipNet protocol holds out 255 utterances per speaker and trains on the rest of the same 33 speakers. `local/make_overlap_split.py` builds it from the unseen-speaker manifests, reusing their features:
```bash
./local/make_overlap_split.py
./conformer_ctc2/train.py \
  --exp-dir conformer_ctc2/exp-overlap \
  --manifest-dir data/avhubert_mf_l8_overlap \
  --num-epochs 30 \
  --dropout 0.1
./conformer_ctc2/decode.py \
  --exp-dir conformer_ctc2/exp-overlap \
  --manifest-dir data/avhubert_mf_l8_overlap \
  --epoch 30 \
  --avg 4
```

|  | WER (overlapped speakers, seed 42) |
|------------------------|-----|
| 1best, 30 epochs, dropout 0.1 | 1.64% |
| 1best, 8 epochs, dropout 0.1 | 2.20% |
| 1best, 8 epochs, dropout 0.4 (the unseen-speaker recipe) | 3.38% |

The heavy regularization that suits unseen speakers underfits seen ones, so this protocol uses the conventional long schedule.

#### Fine-tuned AV-HuBERT front-end

`conformer_ctc2/train_ft.py` trains the same head together with the 8 kept AV-HuBERT transformer layers, starting from the same `base_vox_iter5.pt`; the 3D-conv+ResNet feature extractor stays frozen. The front-end learns at 0.1x the head's learning rate with no weight decay, and 30 s micro-batches are accumulated over 13 steps (390 s per update). It reads the mean-face ROI caches that `prepare.sh` stage 3 writes next to each clip.

|  | WER (unseen speaker, GRID) |
|------------------------|-----|
| 1best, seed 42 | 5.97% |
| 1best, mean of 3 seeds (42/43/44) | 6.25% |

```bash
./conformer_ctc2/train_ft.py --exp-dir conformer_ctc2/exp-ft
./conformer_ctc2/decode.py --finetuned true --exp-dir conformer_ctc2/exp-ft --max-duration 30
```

The seed-42 model (epochs 5 to 8 averaged) is at
<https://huggingface.co/ialmajai/icefall-vsr-grid-conformer-ctc2-phone-l8-ft-2026-10-10>

### 2026-04-30: BPE-58, centroid ROIs, AV-HuBERT layer 9 (superseded)


|  | WER (unseen speaker) |
|------------------------|-----|
| ctc-greedy-search      | 7.35% |
| ctc-decoding           | 7.35% |
| 1best                  | 6.72% |


Check out commit `88c30de1` before running these commands: they rely on that commit's script defaults, and the current defaults train the phone system above instead.

The training command is:
```bash
./conformer_ctc2/train.py \
  --exp-dir conformer_ctc2/exp \
  --max-duration 1400 \
  --use-fp16 1
```

The decoding command is:
```bash
for m in ctc-greedy-search ctc-decoding 1best ; do
  ./conformer_ctc2/decode.py \
    --exp-dir conformer_ctc2/exp \
    --epoch 30 \
    --avg 15 \
    --method $m \
    --use-averaged-model False
done
```

You can find a pretrained model by visiting
<https://huggingface.co/ialmajai/icefall-vsr-grid-conformer-ctc2-bpe-58-2026-04-29>

```bash
# copy pretrained.pt to $PWD/epoch-999.pt

for m in ctc-greedy-search ctc-decoding 1best ; do
  ./conformer_ctc2/decode.py \
    --exp-dir $PWD \
    --epoch 999 \
    --avg 1 \
    --method $m \
    --use-averaged-model False
done
```

