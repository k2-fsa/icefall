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
# matcha

[./matcha](./matcha) contains the code for training [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

This recipe provides a Matcha-TTS model trained on the LJSpeech dataset.

Checkpoints and training logs can be found [here](https://huggingface.co/csukuangfj/icefall-tts-ljspeech-matcha-en-2024-10-28).
The pull-request for this recipe can be found at <https://github.com/k2-fsa/icefall/pull/1773>

The training command is given below:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 ./matcha/train.py \
  --exp-dir ./matcha/exp-new-3/ \
  --num-workers 4 \
  --world-size 4 \
  --num-epochs 4000 \
  --max-duration 1000 \
  --bucketing-sampler 1 \
  --start-epoch 1
```

To inference, use:

```bash
# Download Hifigan vocoder. We use Hifigan v1 below. You can select from v1, v2, or v3

wget https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v1

./matcha/inference \
  --exp-dir ./matcha/exp-new-3 \
  --epoch 4000 \
  --tokens ./data/tokens.txt \
  --vocoder ./generator_v1 \
  --input-text "how are you doing?"
  --output-wav ./generated.wav
```

```bash
soxi ./generated.wav
```
prints:
```
Input File     : './generated.wav'
Channels       : 1
Sample Rate    : 22050
Precision      : 16-bit
Duration       : 00:00:01.29 = 28416 samples ~ 96.6531 CDDA sectors
File Size      : 56.9k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

To export the checkpoint to onnx:

```bash
# export the acoustic model to onnx

./matcha/export_onnx.py \
  --exp-dir ./matcha/exp-new-3 \
  --epoch 4000 \
  --tokens ./data/tokens.txt
```

The above command generate the following files:

  - model-steps-2.onnx
  - model-steps-3.onnx
  - model-steps-4.onnx
  - model-steps-5.onnx
  - model-steps-6.onnx

where the 2 in `model-steps-2.onnx` means it uses 2 steps for the ODE solver.


To export the Hifigan vocoder to onnx, please use:

```bash
wget https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v1
wget https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v2
wget https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v3

python3 ./matcha/export_onnx_hifigan.py
```

The above command generates 3 files:

  - hifigan_v1.onnx
  - hifigan_v2.onnx
  - hifigan_v3.onnx

To use the generated onnx files to generate speech from text, please run:

```bash
python3 ./matcha/onnx_pretrained.py \
 --acoustic-model ./model-steps-6.onnx \
 --vocoder ./hifigan_v1.onnx \
 --tokens ./data/tokens.txt \
 --input-text "Ask not what your country can do for you; ask what you can do for your country." \
 --output-wav ./matcha-epoch-4000-step6-hfigian-v1.wav
```

```bash
soxi ./matcha-epoch-4000-step6-hfigian-v1.wav

Input File     : './matcha-epoch-4000-step6-hfigian-v1.wav'
Channels       : 1
Sample Rate    : 22050
Precision      : 16-bit
Duration       : 00:00:05.46 = 120320 samples ~ 409.252 CDDA sectors
File Size      : 241k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

https://github.com/user-attachments/assets/b7c197a6-3870-49c6-90ca-db4d3776869b

