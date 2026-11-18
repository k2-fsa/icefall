# Introduction

This recipe includes some different ASR models trained with Common Voice

[./RESULTS.md](./RESULTS.md) contains the latest results.

## Japanese Common Voice

Japanese uses the normal Common Voice layout; it is not a separate dataset
recipe. Point `dl_dir` at the parent directory that contains the release
directory, then select the release and language explicitly. For example, the
shared external corpus layout is:

```text
/path/to/commonvoice/cv-corpus-25.0-2026-03-09/ja/{clips,train.tsv,dev.tsv,test.tsv}
```

Run the preparation stages with:

```bash
./prepare.sh \
  --dl-dir /path/to/commonvoice \
  --release cv-corpus-25.0-2026-03-09 \
  --lang ja \
  --stage 1 --stop-stage 9
```

Stage 3 applies NFKC normalization, removes punctuation, and preserves letters,
numbers, combining marks, and collapsed whitespace. The standard ASR path
continues with BPE. The Japanese phone-token KWS recipe is deliberately kept
separate under `egs/commonvoice/KWS`.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                                           |
|---------------------------------------|---------------------|--------------------|---------------------------------------------------|
| `pruned_transducer_stateless7`        | Zipformer           | Embedding + Conv1d | First experiment with Zipformer from Dan          |

The decoder in `transducer_stateless` is modified from the paper
[RNN-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
