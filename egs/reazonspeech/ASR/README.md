# Introduction



**ReazonSpeech** is an open-source dataset that contains a diverse set of natural Japanese speech, collected from terrestrial television streams. It contains more than 35,000 hours of audio.



The dataset is available on Hugging Face. For more details, please visit:

- Dataset: https://huggingface.co/datasets/reazon-research/reazonspeech
- Paper: https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf



[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers



There are various folders containing the name `transducer` in this folder. The following table lists the differences among them.

|                                          | Encoder              | Decoder            | Comment                                           |
| ---------------------------------------- | -------------------- | ------------------ | ------------------------------------------------- |
| `zipformer`                              | Upgraded Zipformer   | Embedding + Conv1d | The latest recipe                                 |

The decoder in `transducer_stateless` is modified from the paper [Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/). We place an additional Conv1d layer right after the input embedding layer.

