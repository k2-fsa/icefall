
# Introduction

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/librispeech.html>
for how to run models in this recipe.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                        | Encoder   | Decoder            |
|------------------------|-----------|--------------------|
| `transducer`           | Conformer | LSTM               |
| `transducer_stateless` | Conformer | Conv1d + Embedding |


