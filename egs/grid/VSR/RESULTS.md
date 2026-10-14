## Results

### [conformer_ctc2](./conformer_ctc2)

### 2026-04-30


|  | WER (unseen speaker) |
|------------------------|-----|
| ctc-greedy-search      | 7.35% |
| ctc-decoding           | 7.35% |
| 1best                  | 6.72% |


The training command using a single NVIDIA GeForce RTX 3080 GPU is:
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

