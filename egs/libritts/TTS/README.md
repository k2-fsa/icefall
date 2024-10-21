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

# VITS

This recipe provides a VITS model trained on the LibriTTS dataset.

The training command is given below:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
./vits/train.py \
  --world-size 4 \
  --num-epochs 1000 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir vits/exp \
  --max-duration 500
```

To inference, use:
```
./vits/infer.py \
  --exp-dir vits/exp \
  --epoch 1000 \
  --tokens data/tokens.txt
```
