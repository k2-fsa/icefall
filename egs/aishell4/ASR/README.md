
# Introduction

This recipe contains some various ASR models trained with Aishell4 (including S, M and L three subsets).

The AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by 8-channel circular microphone array for speech processing in conference scenarios. The dataset consists of 211 recorded meeting sessions, each containing 4 to 8 speakers, with a total length of 120 hours. This dataset aims to bridge the advanced research on multi-speaker processing and the practical application scenario in three aspects. With real recorded meetings, AISHELL-4 provides realistic acoustics and rich natural speech characteristics in conversation such as short pause, speech overlap, quick speaker turn, noise, etc. Meanwhile, the accurate transcription and speaker voice activity are provided for each meeting in AISHELL-4. This allows the researchers to explore different aspects in meeting processing, ranging from individual tasks such as speech front-end processing, speech recognition and speaker diarization, to multi-modality modeling and joint optimization of relevant tasks.

(From [Open Speech and Language Resources](https://www.openslr.org/111/))

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

There are various folders containing the name `transducer` in this folder.
The following table lists the differences among them.

|                                       | Encoder             | Decoder            | Comment                     |
|---------------------------------------|---------------------|--------------------|-----------------------------|
| `pruned_transducer_stateless5`        | Conformer(modified) | Embedding + Conv1d | Using k2 pruned RNN-T loss  |                      |

The decoder in `transducer_stateless` is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
