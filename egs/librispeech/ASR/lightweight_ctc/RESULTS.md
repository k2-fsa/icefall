
## MobileNetS4 with phone lexicon

### 2022-08-12

The number of parameters with phone lexicon is 1766904. The WER with each decoding method is listed as follows. Since the model is targeted for low powered devices, the major goal is to improve one-best WER.

| Decoding method           | test-clean | test-other | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| one-best                  | 11.09      | 28.5       |  |
| nbest-rescoring           | 9.52       | 26.65*     | lm_scale=0.7 gives best WER  |
| whole-lattice-rescoring   | 8.12       | 22.25*     | lm_scale=0.8 gives best WER |
| nbest-oracle              | 6.39       | 20.28      |  |

*: CUDA OOM occured, so effective num_paths/num_arcs may be smaller.

The training command for reproducing is given below:

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

The decoding command is given below:

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

