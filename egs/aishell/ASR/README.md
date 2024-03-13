
# Introduction

Please refer to <https://k2-fsa.github.io/icefall/recipes/Non-streaming-ASR/aishell/index.html> for how to run models in this recipe.

Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co., Ltd.
400 people from different accent areas in China are invited to participate in the recording, which is conducted in a quiet indoor environment using high fidelity microphone and downsampled to 16kHz. The manual transcription accuracy is above 95%, through professional speech annotation and strict quality inspection. The data is free for academic use. We hope to provide moderate amount of data for new researchers in the field of speech recognition.

(From [Open Speech and Language Resources](https://www.openslr.org/33/))

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                    | Encoder   | Decoder            | Comment                                                                           |
|------------------------------------|-----------|--------------------|-----------------------------------------------------------------------------------|
| `transducer_stateless`             | Conformer | Embedding + Conv1d | with `k2.rnnt_loss`                                                               |
| `transducer_stateless_modified`    | Conformer | Embedding + Conv1d | with modified transducer from `optimized_transducer`                     |
| `transducer_stateless_modified-2`  | Conformer | Embedding + Conv1d | with modified transducer from `optimized_transducer` + extra data      |
| `pruned_transducer_stateless3`     | Conformer (reworked) | Embedding + Conv1d | pruned RNN-T + reworked model with random combiner + using aidatatang_20zh as extra data|
| `pruned_transducer_stateless7`     | Zipformer | Embedding | pruned RNN-T + zipformer encoder + stateless decoder with context-size set to 1 |
| `zipformer`                           | Upgraded Zipformer | Embedding + Conv1d | The latest recipe with context-size set to 1 |


The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.

# Whisper

Recipe to finetune large pretrained models
|                                    | Encoder   | Decoder            | Comment                                                                           |
|------------------------------------|-----------|--------------------|-----------------------------------------------------------------------------------|
| `whisper`             | Transformer | Transformer | support fine-tuning using deepspeed
