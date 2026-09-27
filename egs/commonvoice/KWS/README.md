# Japanese Common Voice KWS

This is an initial open-vocabulary Japanese keyword-spotting training recipe.
It trains a small causal Zipformer Transducer on Japanese phone tokens, then a
keyword graph can be added at decoding time. It does **not** train a closed-set
command classifier and does not use command fine-tuning.

The recipe is intentionally separate from `../ASR`: Common Voice preparation
and its Japanese normalization are shared concepts, but the model units,
streaming configuration, and eventual KWS evaluation protocol are different.

## Input and output contract

`prepare.sh` accepts a directory that is directly the Japanese Common Voice
directory, i.e. it contains `clips/`, `train.tsv`, `dev.tsv`, and `test.tsv`.
Raw audio is read in place and is never copied into the checkout.

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
claim. The follow-up KWS evaluation must report recall, false alarms per hour,
and trigger latency on a fixed held-out audio set.

## Phone error rate

After training, measure phone recognition on the held-out Common Voice test
split before tuning the keyword graph:

```bash
./evaluate.sh \
  --data-dir /workspace/work/japanese-open-kws/cv25-ja/data \
  --exp-dir /workspace/artifacts/japanese-open-kws/cv25-ja \
  --checkpoint best-valid-loss.pt
```

The decoder writes both the official-test PER and a second PER after removing
the same implausible transcript/audio-rate outliers used by training and
validation. PER is an acoustic/token-sequence sanity metric; it does not replace
keyword recall, false alarms per hour, or trigger-latency evaluation.

## Phone frontend

Text is first normalized using the Japanese Common Voice normalizer from the
parent branch. `pyopenjtalk.g2p(..., kana=False)` supplies the phone sequence;
boundary `sil` and `pau` symbols are removed while lexical closure `cl` is
retained. `tokens.txt` is learned from the normalized training transcripts only
and includes `<blk>` and `<unk>`.

The launcher image must provide the `pyopenjtalk-plus` drop-in. Run
`runpod/doctor.sh` in the operations repository before preparation so that an
image build or a Pod `Running` state is never mistaken for a usable CUDA/k2
runtime.
