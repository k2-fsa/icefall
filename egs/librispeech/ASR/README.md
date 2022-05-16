# Introduction

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/librispeech/index.html> for how to run models in this recipe.

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                                           |
|---------------------------------------|---------------------|--------------------|---------------------------------------------------|
| `transducer`                          | Conformer           | LSTM               |                                                   |
| `transducer_stateless`                | Conformer           | Embedding + Conv1d | Using optimized_transducer from computing RNN-T loss  |
| `transducer_stateless2`               | Conformer           | Embedding + Conv1d | Using torchaudio for computing RNN-T loss             |
| `transducer_lstm`                     | LSTM                | LSTM               |                                                   |
| `transducer_stateless_multi_datasets` | Conformer           | Embedding + Conv1d | Using data from GigaSpeech as extra training data |
| `pruned_transducer_stateless`         | Conformer           | Embedding + Conv1d | Using k2 pruned RNN-T loss                        |
| `pruned_transducer_stateless2`        | Conformer(modified) | Embedding + Conv1d | Using k2 pruned RNN-T loss                        |
| `pruned_transducer_stateless3`        | Conformer(modified) | Embedding + Conv1d | Using k2 pruned RNN-T loss + using GigaSpeech as extra training data |


The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
