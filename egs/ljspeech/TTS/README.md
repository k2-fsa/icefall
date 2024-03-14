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

## Quality vs speed

If you feel that the trained model is slow at runtime, you can specify the
argument `--model-type` during training. Possible values are:

  - `low`, means **low** quality. The resulting model is very small in file size
    and runs very fast. The following is a wave file generatd by a `low` quality model

    https://github.com/k2-fsa/icefall/assets/5284924/d5758c24-470d-40ee-b089-e57fcba81633

    The text is `Ask not what your country can do for you; ask what you can do for your country.`

    The exported onnx model has a file size of ``26.8 MB`` (float32).

  - `medium`, means **medium** quality.
    The following is a wave file generatd by a `medium` quality model

    https://github.com/k2-fsa/icefall/assets/5284924/b199d960-3665-4d0d-9ae9-a1bb69cbc8ac

    The text is `Ask not what your country can do for you; ask what you can do for your country.`

    The exported onnx model has a file size of ``70.9 MB`` (float32).

  - `high`, means **high** quality. This is the default value.

    The following is a wave file generatd by a `high` quality model

    https://github.com/k2-fsa/icefall/assets/5284924/b39f3048-73a6-4267-bf95-df5abfdb28fc

    The text is `Ask not what your country can do for you; ask what you can do for your country.`

    The exported onnx model has a file size of ``113 MB`` (float32).


A pre-trained `low` model trained using 4xV100 32GB GPU with the following command can be found at
<https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-low-2024-03-12>

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
./vits/train.py \
  --world-size 4 \
  --num-epochs 1601 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir vits/exp \
  --model-type low \
  --max-duration 800
```

A pre-trained `medium` model trained using 4xV100 32GB GPU with the following command can be found at
<https://huggingface.co/csukuangfj/icefall-tts-ljspeech-vits-medium-2024-03-12>
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
./vits/train.py \
  --world-size 4 \
  --num-epochs 1000 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir vits/exp-medium \
  --model-type medium \
  --max-duration 500

# (Note it is killed after `epoch-820.pt`)
```
