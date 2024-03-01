# Introduction

This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. 
A transcription is provided for each clip. 
Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

The texts were published between 1884 and 1964, and are in the public domain. 
The audio was recorded in 2016-17 by the [LibriVox](https://librivox.org/) project and is also in the public domain.

The above information is from the [LJSpeech website](https://keithito.com/LJ-Speech-Dataset/).

# VITS

This recipe provides a VITS model trained on the LJSpeech dataset.

Pretrained model can be found [here](https://huggingface.co/Zengwei/icefall-tts-ljspeech-vits-2024-02-28).

For tutorial and more details, please refer to the [VITS documentation](https://k2-fsa.github.io/icefall/recipes/TTS/ljspeech/vits.html).

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