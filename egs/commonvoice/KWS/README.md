# Japanese Common Voice KWS

This is an initial open-vocabulary Japanese keyword-spotting training recipe.
It trains a small causal Zipformer Transducer on Japanese phone tokens, then a
keyword graph can be added at decoding time. It does **not** train a closed-set
command classifier and does not use command fine-tuning.

The recipe is intentionally separate from `../ASR`: Common Voice preparation
and its Japanese normalization are shared concepts, but the model units,
streaming configuration, and KWS evaluation protocol are different.

## Input and output contract

`prepare.sh` accepts a directory that is directly the Japanese Common Voice
directory, i.e. it contains `clips/`, `train.tsv`, `dev.tsv`, and `test.tsv`.
Raw audio is read in place and is never copied into the checkout.
Each `data-dir` is tied to one resolved Common Voice root; use a new data
directory when changing the source path.

```bash
./prepare.sh \
  --commonvoice-root /path/to/cv-corpus-25.0-2026-03-09/ja \
  --data-dir /workspace/work/japanese-open-kws/cv25-ja/data

./train.sh \
  --data-dir /workspace/work/japanese-open-kws/cv25-ja/data \
  --exp-dir /workspace/artifacts/japanese-open-kws/cv25-ja \
  --num-epochs 1
```

Preparation writes Lhotse manifests, 80-bin fbank features, and
`lang_phone/tokens.txt` beneath `data-dir`. The source audio paths in the
manifests point back to `commonvoice-root`.

`train.sh` defaults to one GPU and a small causal Zipformer configuration
derived from the maintained WenetSpeech KWS recipe.
It writes `runtime.json` after a successful run; its `wall_seconds` includes
training and the end-of-epoch validation pass, making it the value to use for
the first GPU-credit estimate.

One epoch is a wiring and throughput measurement only. It is not a quality
claim. The KWS evaluation below reports recall and false alarms per hour on a
fixed held-out audio set. Streaming trigger latency needs a separate test.

## Phone error rate

After training, measure phone recognition on the held-out Common Voice test
split before tuning the keyword graph:

```bash
./evaluate.sh \
  --data-dir /workspace/work/japanese-open-kws/cv25-ja/data \
  --exp-dir /workspace/artifacts/japanese-open-kws/cv25-ja \
  --split dev \
  --epoch 30 \
  --avg 15 \
  --use-averaged-model true
```

The decoder writes both the selected split's official PER and a second PER
after removing the same implausible transcript/audio-rate outliers used by
training and validation. PER is an acoustic/token-sequence sanity metric; it
does not replace keyword recall, false alarms per hour, or trigger-latency
evaluation.

Tune averaging, search, and blank penalty on `dev`, not `test`. To resume a
completed experiment while retaining its optimizer, scheduler, and averaged
model state:

```bash
./train.sh \
  --data-dir /workspace/work/japanese-open-kws/cv25-ja/data \
  --exp-dir /workspace/artifacts/japanese-open-kws/cv25-ja \
  --start-epoch 31 \
  --num-epochs 50
```

## Keyword spotting evaluation

The evaluation follows the WenetSpeech KWS recipe's use of a fixed keyword
list, positive recordings, and negative audio duration. The 20 phrases in
`eval_keywords.txt` were fixed using development transcript counts and phone
collision diagnostics before examining any test predictions. Three short
words in an initial development list were replaced because their phone
sequences occur within common unrelated words (for example, the phones for
`明日` are a suffix of `ました`). This is a **read-speech phrase spotting** test:
the target phrases occur inside Common Voice sentences. It is not a wake-word,
natural command, noisy microphone, or streaming benchmark. Short Japanese
words and homophones are especially difficult for a phone-only model.

The preparer uses normalized transcript substring matches as labels, excludes
clips whose normalized transcript exceeds 20 characters per audio second, and
keeps all remaining clips. A clip with no target phrase contributes its full
duration to the negative-audio denominator. Its summary also counts positive
labels whose isolated keyword phones do not appear verbatim in the full
sentence's phone sequence, and negative clips where a keyword phone sequence
appears without the matching written word. Those are label limitations, not
model errors. The scorer counts the first hit for each correct phrase in a
positive clip as one true positive; duplicate hits and wrong phrases are false
positives. It reports negative false-alarm
**events** per hour and negative clips with any alarm per hour separately.
The former is closest to the WenetSpeech convention. Inspect `per_keyword`
results as well as the aggregate: the keyword set changes the difficulty.

The following commands regenerate the private manifests and score one raw
checkpoint. Paths under `/path/to/private-results` must stay outside the Git
checkout. Use the same decoder configuration for `dev` and `test`.

```bash
python3 local/prepare_kws_eval.py \
  --commonvoice-root /path/to/cv-corpus-25.0-2026-03-09/ja \
  --keywords-file eval_keywords.txt \
  --output-dir /path/to/private-results

python3 zipformer/evaluate_kws.py \
  --manifest /path/to/private-results/dev.jsonl \
  --commonvoice-root /path/to/cv-corpus-25.0-2026-03-09/ja \
  --keywords-file eval_keywords.txt \
  --tokens /path/to/data/lang_phone/tokens.txt \
  --exp-dir /path/to/experiment \
  --checkpoint epoch-60.pt \
  --output-jsonl /path/to/private-results/dev-predictions.jsonl

python3 local/score_kws_eval.py \
  --manifest /path/to/private-results/dev.jsonl \
  --predictions /path/to/private-results/dev-predictions.jsonl \
  --keywords-file eval_keywords.txt \
  --min-ac-prob 0.35 \
  --output-json /path/to/private-results/dev-score.json
```

Run the same decoder with `test.jsonl` and a new output path, then score its
output using the acoustic-probability post-filter chosen on `dev`. The graph
threshold (`--keywords-threshold`) is a lower bound for that post-filter, so
the latter can be swept on a fixed decoder output without decoding the audio
again. This post-filter sweep is not equivalent to rerunning the stateful
decoder with a different graph threshold. Predictions contain clip IDs,
keyword hits, and approximate times; do not publish them or Common Voice audio
with the model. Publish the fixed keyword list, code, source TSV hashes,
denominators, selected settings, and aggregate results. See `RESULTS.md` for
the measured operating point.

### Streaming ONNX and sherpa-onnx

The maintained WenetSpeech KWS exporter can export this model architecture.
Use a dedicated private export directory containing a checkpoint named
`epoch-60.pt` and the matching `tokens.txt`; only the resulting ONNX models,
tokens, and generated keyword file belong in a model package. From this recipe
directory (with `onnx` and `onnxruntime` installed):

```bash
PYTHONPATH=../../.. python3 ../../wenetspeech/KWS/zipformer/export-onnx-streaming.py \
  --exp-dir /path/to/export-dir \
  --tokens /path/to/export-dir/tokens.txt \
  --epoch 60 --avg 1 --use-averaged-model false \
  --enable-int8-quantization 0 \
  --chunk-size 16 --left-context-frames 64 \
  --decoder-dim 320 --joiner-dim 320 \
  --num-encoder-layers 1,1,1,1,1,1 \
  --feedforward-dim 192,192,192,192,192,192 \
  --encoder-dim 128,128,128,128,128,128 \
  --encoder-unmasked-dim 128,128,128,128,128,128 \
  --causal true

python3 local/make_sherpa_keywords.py \
  --keywords-file eval_keywords.txt \
  --tokens /path/to/export-dir/tokens.txt \
  --output /path/to/export-dir/keywords.txt

python3 zipformer/evaluate_sherpa_onnx.py \
  --manifest /path/to/private-results/test.jsonl \
  --commonvoice-root /path/to/cv-corpus-25.0-2026-03-09/ja \
  --model-dir /path/to/export-dir \
  --keywords-file /path/to/export-dir/keywords.txt \
  --output-jsonl /path/to/private-results/test-sherpa-predictions.jsonl
```

The final command requires the optional `sherpa-onnx` Python package and feeds
0.16-second audio chunks to its online KeywordSpotter. Score its predictions
with the same `local/score_kws_eval.py`, without the acoustic-probability
post-filter. The keyword file contains phone tokens and `@`-prefixed display
text; arbitrary new Japanese keywords require the same phone frontend.

## Phone frontend

Text is first normalized using the Japanese Common Voice normalizer from the
parent branch. `pyopenjtalk.g2p(..., kana=False)` supplies the phone sequence;
boundary `sil` and `pau` symbols are removed while lexical closure `cl` is
retained. `tokens.txt` is learned from the normalized training transcripts only
and includes `<blk>` and `<unk>`.

The runtime must provide `pyopenjtalk-plus` (imported as `pyopenjtalk`) and
Lhotse's `lhotse` console command. Before preparation, verify both with
`python3 -c 'import pyopenjtalk, lhotse'` and `lhotse --help`. GPU training also
requires a working CUDA PyTorch and k2 installation.
