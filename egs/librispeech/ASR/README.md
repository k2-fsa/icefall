# Introduction

Please refer to <https://k2-fsa.github.io/icefall/recipes/Non-streaming-ASR/librispeech/index.html> for how to run models in this recipe.

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
| `pruned_transducer_stateless4`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless2 + save averaged models periodically during training + delay penalty |
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless4 + more layers + random combiner|
| `pruned_transducer_stateless6`        | Conformer(modified) | Embedding + Conv1d | same as pruned_transducer_stateless4 + distillation with hubert|
| `pruned_transducer_stateless7`        | Zipformer | Embedding + Conv1d | First experiment with Zipformer from Dan|
| `pruned_transducer_stateless7_ctc`    | Zipformer | Embedding + Conv1d | Same as pruned_transducer_stateless7, but with extra CTC head|
| `pruned_transducer_stateless7_ctc_bs` | Zipformer | Embedding + Conv1d | pruned_transducer_stateless7_ctc + blank skip |
| `pruned_transducer_stateless7_streaming` | Streaming Zipformer | Embedding + Conv1d | streaming version of pruned_transducer_stateless7 |
| `pruned_transducer_stateless7_streaming_multi` | Streaming Zipformer | Embedding + Conv1d | same as pruned_transducer_stateless7_streaming, trained on LibriSpeech + GigaSpeech  |
| `pruned_transducer_stateless8`        | Zipformer | Embedding + Conv1d | Same as pruned_transducer_stateless7, but using extra data from GigaSpeech|
| `pruned_stateless_emformer_rnnt2`     | Emformer(from torchaudio) | Embedding + Conv1d | Using Emformer from torchaudio for streaming ASR|
| `conv_emformer_transducer_stateless`  | ConvEmformer | Embedding + Conv1d | Using ConvEmformer for streaming ASR + mechanisms in reworked model |
| `conv_emformer_transducer_stateless2` | ConvEmformer | Embedding + Conv1d | Using ConvEmformer with simplified memory for streaming ASR + mechanisms in reworked model |
| `lstm_transducer_stateless`           | LSTM | Embedding + Conv1d | Using LSTM with mechanisms in reworked model |
| `lstm_transducer_stateless2`          | LSTM | Embedding + Conv1d | Using LSTM with mechanisms in reworked model + gigaspeech (multi-dataset setup) |
| `lstm_transducer_stateless3`          | LSTM | Embedding + Conv1d | Using LSTM with mechanisms in reworked model + gradient filter + delay penalty |
| `zipformer`                           | Upgraded Zipformer | Embedding + Conv1d | The latest recipe |
| `zipformer_adapter`                           | Upgraded Zipformer | Embedding + Conv1d | It supports domain adaptation of Zipformer using parameter efficient adapters |
| `zipformer_adapter`                           | Upgraded Zipformer | Embedding + Conv1d | Finetune Zipformer with LoRA  |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.

# CTC

|                              | Encoder            | Comment                      |
|------------------------------|--------------------|------------------------------|
| `conformer-ctc`              | Conformer          | Use auxiliary attention head |
| `conformer-ctc2`             | Reworked Conformer | Use auxiliary attention head |
| `conformer-ctc3`             | Reworked Conformer | Streaming version + delay penalty |
| `zipformer-ctc`              | Zipformer          | Use auxiliary attention head |
| `zipformer`                  | Upgraded Zipformer | Use auxiliary transducer head / attention-decoder head (the latest recipe) |

# MMI

|                              | Encoder   | Comment                                           |
|------------------------------|-----------|---------------------------------------------------|
| `conformer-mmi`              | Conformer |                                                   |
| `zipformer-mmi`              | Zipformer | CTC warmup + use HP as decoding graph for decoding |

# CR-CTC

|                              | Encoder            | Comment                      |
|------------------------------|--------------------|------------------------------|
| `zipformer`                  | Upgraded Zipformer | Could also be an auxiliary loss to improve transducer or CTC/AED (the latest recipe) |
