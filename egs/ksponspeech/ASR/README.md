# Introduction
KsponSpeech is a large-scale spontaneous speech corpus of Korean.
This corpus contains 969 hours of open-domain dialog utterances,
spoken by about 2,000 native Korean speakers in a clean environment.

All data were constructed by recording the dialogue of two people
freely conversing on a variety of topics and manually transcribing the utterances.

The transcription provides a dual transcription consisting of orthography and pronunciation,
and disfluency tags for spontaneity of speech, such as filler words, repeated words, and word fragments.

The original audio data has a pcm extension.
During preprocessing, it is converted into a file in the flac extension and saved anew.

KsponSpeech is publicly available on an open data hub site of the Korea government.
The dataset must be downloaded manually.

For more details, please visit:

 - Dataset: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
 - Paper: https://www.mdpi.com/2076-3417/10/19/6936

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers
There are various folders containing the name `transducer` in this folder. The following table lists the differences among them.

|                                          | Encoder              | Decoder            | Comment                                           |
| ---------------------------------------- | -------------------- | ------------------ | ------------------------------------------------- |
| `pruned_transducer_stateless7_streaming` | Streaming Zipformer  | Embedding + Conv1d | streaming version of pruned_transducer_stateless7 |
| `zipformer`                              | Upgraded Zipformer   | Embedding + Conv1d | The latest recipe                                 |

The decoder in `transducer_stateless` is modified from the paper [Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/). We place an additional Conv1d layer right after the input embedding layer.