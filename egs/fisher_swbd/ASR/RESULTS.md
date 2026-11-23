## Results

### SWBD BPE training results (Conformer-CTC)

#### 01-17-2022

This recipe is based on LibriSpeech. 
Data preparation/normalization is a simplified version of the one found in Kaldi.
The data is resampled to 16kHz on-the-fly -- it's not needed, but makes it easier to combine with other corpora,
and likely doesn't affect the results too much.
The training set was only Switchboard, minus 20 held-out conversations (dev data, ~1h of speech).
This was tested only on the dev data.
We didn't tune the model, hparams, or language model in any special way vs. LibriSpeech recipe.
No rescoring was used (decoding method: "1best").
The model was trained on a single A100 GPU (24GB RAM) for 2 days.

WER (it includes `[LAUGHTER]`, `[NOISE]`, `[VOCALIZED-NOISE]` so the "real" WER is likely lower):

10 epochs (avg 5) : 19.58%
20 epochs (avg 10): 12.61%
30 epochs (avg 20): 11.24%
35 epochs (avg 20): 10.96%
40 epochs (avg 20): 10.94%

To reproduce the above result, use the following commands for training:

```
cd egs/librispeech/ASR/conformer_ctc
./prepare.sh --swbd-only true
export CUDA_VISIBLE_DEVICES="0"
./conformer_ctc/train.py \
  --lr-factor 1.25 \
  --max-duration 200 \
  --num-workers 14 \
  --lang-dir data/lang_bpe_500 \
  --num-epochs 40
```

and the following command for decoding

```
python conformer_ctc/decode.py \
  --epoch 40 \
  --avg 20 \
  --method 1best
```

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/0mvXl9BYRJ62J1fVnILm0w/>
