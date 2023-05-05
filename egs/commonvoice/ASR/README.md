# Introduction

This recipe includes some different ASR models trained with Common Voice

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                                           |
|---------------------------------------|---------------------|--------------------|---------------------------------------------------|
| `pruned_transducer_stateless7`        | Zipformer           | Embedding + Conv1d | First experiment with Zipformer from Dan          |

The decoder in `transducer_stateless` is modified from the paper
[RNN-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
