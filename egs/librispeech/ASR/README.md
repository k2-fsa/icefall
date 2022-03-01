
# Introduction

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/librispeech.html>
for how to run models in this recipe.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder   | Decoder            | Comment                                           |
|---------------------------------------|-----------|--------------------|---------------------------------------------------|
| `transducer`                          | Conformer | LSTM               |                                                   |
| `transducer_stateless`                | Conformer | Embedding + Conv1d |                                                   |
| `transducer_lstm`                     | LSTM      | LSTM               |                                                   |
| `transducer_stateless_multi_datasets` | Conformer | Embedding + Conv1d | Using data from GigaSpeech as extra training data |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
