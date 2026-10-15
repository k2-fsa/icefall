# Neural contextual biasing for the Zipformer transducer

This recipe adds neural contextual biasing to the Zipformer transducer of
[pruned_transducer_stateless7](../pruned_transducer_stateless7): given a list
of biasing words for an utterance (e.g., names or rare words), the model
attends to embeddings of these words to recognize them more accurately.

The ASR model is a pretrained, frozen pruned_transducer_stateless7 model;
only the biasing modules are trained. The setup follows the LibriSpeech
biasing benchmark of [Le et al., 2021](#references) and implements the neural
biasing baseline of
[Improving Neural Biasing for Contextual Speech Recognition by Early Context Injection and Text Perturbation](https://arxiv.org/abs/2407.10303).
The early context injection and text perturbation methods of that paper are
not part of this recipe yet.

## How it works

- **Context encoder** ([context_encoder_lstm.py](./context_encoder_lstm.py)):
  each biasing word is split into BPE tokens and embedded with a bidirectional
  LSTM. Alternatives: pretrained fastText or BERT word embeddings
  (`--is-pretrained-context-encoder true`), or the transducer decoder followed
  by an LSTM (`--is-reused-context-encoder true`).
- **Biasing modules** ([biasing_module.py](./biasing_module.py)): the encoder
  output and the decoder (prediction network) output attend to the word
  embeddings plus an all-zero "no-bias" embedding; the attention outputs are
  added to them ([model.py](./model.py)).
- **Training biasing lists** ([context_collector.py](./context_collector.py)):
  the "rare" words of each utterance (words not among the 5k most frequent
  training words) plus `--n-distractors` random rare words.
- **Evaluation**: the predefined biasing lists of size N = 100, 500, 1000 or
  2000 for test-clean and test-other, scored with WER, U-WER (unbiased words)
  and B-WER (biased words) by [score.py](./score.py), taken from
  [fbai-speech](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias).
- **Optional WFST biasing** ([context_wfst.py](./context_wfst.py),
  [biased_lm.py](./biased_lm.py)): shallow fusion with a WFST built from the
  biasing list, in `modified_beam_search`.

## Data preparation

Run the following from `egs/librispeech/ASR`.

1. Features of LibriSpeech and MUSAN (stages 0-4 of `prepare.sh`):

   ```bash
   ./prepare.sh --stage 0 --stop-stage 4
   ```

2. The pretrained pruned_transducer_stateless7 model and its BPE model. Use
   this `bpe.model`, not one trained by `prepare.sh`, since token IDs must
   match the pretrained model. If `data/lang_bpe_500` already exists, e.g.,
   from another recipe, save it elsewhere instead and pass that path to
   `--bpe-model`:

   ```bash
   repo=https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/resolve/main
   dir=icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
   mkdir -p $dir/exp data/lang_bpe_500
   curl -L -o $dir/exp/pretrained.pt $repo/exp/pretrained.pt
   curl -L -o data/lang_bpe_500/bpe.model $repo/data/lang_bpe_500/bpe.model
   ```

3. The word lists and predefined biasing lists:

   ```bash
   repo=https://raw.githubusercontent.com/facebookresearch/fbai-speech/main/is21_deep_bias
   dir=data/fbai-speech/is21_deep_bias
   mkdir -p $dir/words $dir/ref
   for f in all_rare_words.txt common_words_5k.txt; do
     curl -L -o $dir/words/$f $repo/words/$f
   done
   for s in test-clean test-other; do
     for n in 100 500 1000 2000; do
       curl -L -o $dir/ref/$s.biasing_$n.tsv $repo/ref/$s.biasing_$n.tsv
     done
   done
   ```

## Testing

A quick CPU test with a tiny model and fake word lists, which needs none of
the data above:

```bash
python ./pruned_transducer_stateless7_contextual/test_model.py
```

## Training

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_contextual/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --full-libri 1 \
  --max-duration 1600 \
  --exp-dir pruned_transducer_stateless7_contextual/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --init-asr-ckpt icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained.pt \
  --context-dir data/fbai-speech/is21_deep_bias \
  --n-distractors 100
```

`--init-asr-ckpt` initializes the frozen ASR model when training starts from
scratch; the biasing modules are initialized randomly. To resume training,
pass `--start-epoch` (or `--start-batch`) as usual.

Biasing-related options:

| Option | Default | Description |
|---|---|---|
| `--context-dim` | 128 | Size of the word embeddings and of the biasing attention. Pass the same value to `decode.py`. |
| `--n-distractors` | 100 | Random distractors per training utterance; -1 for a random number in [10, 500). |
| `--keep-ratio` | 1.0 | Probability of keeping each rare word of an utterance in its list, to simulate incomplete lists. |
| `--is-full-context` | false | Put all words of an utterance into its list, not only the rare ones. |
| `--asr-eval-mode` | false | Keep the frozen ASR model in eval mode (no dropout or layer skipping) while training the biasing modules. |
| `--is-pretrained-context-encoder` | false | Use pretrained word embeddings: `--pretrained-word-encoder fasttext` (with `--fasttext-embeddings` and `--fasttext-model`) or `bert`. Pass the same options to `decode.py`. |
| `--is-reused-context-encoder` | false | Embed the words with the transducer decoder followed by an LSTM. |

## Decoding

Decode with the predefined biasing lists of size N = 100:

```bash
./pruned_transducer_stateless7_contextual/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir pruned_transducer_stateless7_contextual/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --context-dir data/fbai-speech/is21_deep_bias \
  --is-predefined true \
  --n-distractors 100 \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --max-duration 600
```

For each test set, a line `WER(U-WER/B-WER)` is printed at the end; recognition results and error statistics are written to
`<exp-dir>/modified_beam_search/`.

- Use `--n-distractors 500`, `1000` or `2000` for the larger lists.
- For a baseline without biasing, add
  `--no-encoder-biasing true --no-decoder-biasing true`.
- WFST biasing is enabled with `--no-wfst-lm-biasing false` and a positive
  `--biased-lm-scale`.
- Decoder-side and WFST biasing are only implemented in
  `modified_beam_search` and `modified_beam_search_LODR`; other decoding
  methods only support encoder-side biasing.

## Export

[export.py](./export.py) averages checkpoints into a single `pretrained.pt`;
see its docstring for usage. TorchScript and ONNX export are not supported
yet.

## References

- D. Le et al., "Contextualized Streaming End-to-End Speech Recognition with
  Trie-Based Deep Biasing and Shallow Fusion", Interspeech 2021.
  Data and scoring: https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias
- "Improving Neural Biasing for Contextual Speech Recognition by Early
  Context Injection and Text Perturbation", https://arxiv.org/abs/2407.10303
