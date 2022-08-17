
## MobileNetS4 with phone lexicon

### 2022-08-12

The number of parameters with phone lexicon is 1766904. The WER for each decoding method is listed as follows. Since the model is targeted for low powered devices, the major goal is to improve one-best WER.

| Decoding method           | test-clean | test-other | comment                      |
|---------------------------|------------|------------|------------------------------|
| one-best                  | 11.1       | 28.49      |                              |
| nbest-rescoring           | 8.78       | 24.94      | lm_scale=0.7                 |
| whole-lattice-rescoring   | 8.12       | 22.3*      | lm_scale=0.8                 |
| nbest-oracle              | 5.87       | 19.32      |                              |

*: CUDA OOM occured during decoding.

Pretrained model to can be downloaded here: https://huggingface.co/wangtiance/lightweight_ctc.
Use it to replicate above results:

```bash
ln -s pretrained.pt epoch-9999.pt

python lightweight_ctc/decode.py \
  --epoch 9999 \
  --avg 1 \
  --use-averaged-model 0 \
  --lang-dir data/lang_phone \
  --exp-dir lightweight_ctc/exp_phone \
  --max-duration 100 \
  --num-paths 50 \
  --nbest-scale 1.0
```

The training command is given below:

```bash
cd egs/librispeech/ASR/
./prepare.sh

python lightweight_ctc/train.py \
  --num-epochs 20 \
  --start-epoch 0 \
  --lang-dir data/lang_phone
  --exp-dir lightweight_ctc/exp_phone \
  --full-libri 1 \
  --max-duration 600 \
```

Decoding command is given below:

```bash
python lightweight_ctc/decode.py \
  --epoch 17 \
  --avg 2 \
  --use-averaged-model 1 \
  --lang-dir data/lang_phone \
  --exp-dir lightweight_ctc/exp_phone \
  --max-duration 50 \
  --num-paths 50 \
```

