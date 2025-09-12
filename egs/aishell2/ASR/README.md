
# Introduction

This recipe contains various different ASR models trained with Aishell2.

In AISHELL-2, 1000 hours of clean read-speech data from iOS is published, which is free for academic usage. On top of AISHELL-2 corpus, an improved recipe is developed and released, containing key components for industrial applications, such as Chinese word segmentation, flexible vocabulary expension and phone set transformation etc. Pipelines support various state-of-the-art techniques, such as time-delayed neural networks and Lattic-Free MMI objective funciton. In addition, we also release dev and test data from other channels (Android and Mic).

(From [AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale](https://arxiv.org/abs/1808.10583))

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                     |
|---------------------------------------|---------------------|--------------------|-----------------------------|
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless5 in librispeech recipe  |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
