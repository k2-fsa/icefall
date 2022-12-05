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
| `pruned_transducer_stateless4`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless2 + save averaged models periodically during training                        |
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless4 + more layers + random combiner|
| `pruned_transducer_stateless6`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless4 + distillation with hubert|
| `pruned_transducer_stateless7`        | Zipformer | Embedding + Conv1d | First experiment with Zipformer from Dan|
| `pruned_transducer_stateless7_ctc`    | Zipformer | Embedding + Conv1d | Same as pruned_transducer_stateless7, but with extra CTC head|
| `pruned_transducer_stateless8`        | Zipformer | Embedding + Conv1d | Same as pruned_transducer_stateless7, but using extra data from GigaSpeech|
| `pruned_stateless_emformer_rnnt2`     | Emformer(from torchaudio) | Embedding + Conv1d | Using Emformer from torchaudio for streaming ASR|
| `conv_emformer_transducer_stateless`  | ConvEmformer | Embedding + Conv1d | Using ConvEmformer for streaming ASR + mechanisms in reworked model |
| `conv_emformer_transducer_stateless2` | ConvEmformer | Embedding + Conv1d | Using ConvEmformer with simplified memory for streaming ASR + mechanisms in reworked model |
| `lstm_transducer_stateless`           | LSTM | Embedding + Conv1d | Using LSTM with mechanisms in reworked model |
| `lstm_transducer_stateless2`           | LSTM | Embedding + Conv1d | Using LSTM with mechanisms in reworked model + gigaspeech (multi-dataset setup) |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
