
# Introduction

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/aishell.html>
for how to run models in this recipe.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                    | Encoder   | Decoder            | Comment                                                                           |
|------------------------------------|-----------|--------------------|-----------------------------------------------------------------------------------|
| `transducer_stateless`             | Conformer | Embedding + Conv1d | with `k2.rnnt_loss`                                                               |
| `transducer_stateless_modified`    | Conformer | Embedding + Conv1d | with modified transducer from `optimized_transducer`                     |
| `transducer_stateless_modified-2`  | Conformer | Embedding + Conv1d | with modified transducer from `optimized_transducer` + extra data      |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
