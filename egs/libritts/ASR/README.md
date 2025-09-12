# Introduction

LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. 
The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus. 
The main differences from the LibriSpeech corpus are listed below:
1. The audio files are at 24kHz sampling rate.
2. The speech is split at sentence breaks.
3. Both original and normalized texts are included.
4. Contextual information (e.g., neighbouring sentences) can be extracted.
5. Utterances with significant background noise are excluded.
For more information, refer to the paper "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019. If you use the LibriTTS corpus in your work, please cite this paper where it was introduced.


This recipe includes some different ASR models trained with [LibriTTS](https://openslr.org/60/).

[./RESULTS.md](./RESULTS.md) contains the latest results.

# Transducers

|                                       | Encoder             | Decoder            |
|---------------------------------------|---------------------|--------------------|
| `zipformer`                           | Upgraded Zipformer | Embedding + Conv1d |

The decoder is modified from the paper
[Rnn-Transducer with Stateless Prediction Network](https://ieeexplore.ieee.org/document/9054419/).
We place an additional Conv1d layer right after the input embedding layer.
