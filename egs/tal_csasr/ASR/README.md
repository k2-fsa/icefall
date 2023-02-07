
# Introduction

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/Non-streaming-ASR/librispeech/index.html> for how to run models in this recipe.

This recipe includes some different ASR models trained with TAL_CSASR.

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                     |
|---------------------------------------|---------------------|--------------------|-----------------------------|
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless4 + more layers + random combiner|
| `conv_emformer_transducer_stateless2` | ConvEmformer | Embedding + Conv1d | Using ConvEmformer with simplified memory for streaming ASR + mechanisms in reworked model |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
