# Introduction

It is for the dataset from
https://en.data-baker.com/datasets/freeDatasets/

The dataset contains 10000 Chinese sentences of a native Chinese female speaker,
which is about 12 hours.


**Note**: The dataset is for non-commercial use only.


# matcha

[./matcha](./matcha) contains the code for training [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

Checkpoints and training logs can be found [here](https://huggingface.co/csukuangfj/icefall-tts-baker-matcha-zh-2024-12-27).
The pull-request for this recipe can be found at <https://github.com/k2-fsa/icefall/pull/1849>

The training command is given below:
```bash
python3 ./matcha/train.py \
  --exp-dir ./matcha/exp-1/ \
  --num-workers 4 \
  --world-size 1 \
  --num-epochs 2000 \
  --max-duration 1200 \
  --bucketing-sampler 1 \
  --start-epoch 1
```

To inference, use:

```bash
# Download Hifigan vocoder. We use Hifigan v2 below. You can select from v1, v2, or v3

wget https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v2

python3 ./matcha/infer.py \
  --epoch 2000 \
  --exp-dir ./matcha/exp-1 \
  --vocoder ./generator_v2 \
  --tokens ./data/tokens.txt \
  --cmvn ./data/fbank/cmvn.json \
  --input-text "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。" \
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
Duration       : 00:00:17.31 = 381696 samples ~ 1298.29 CDDA sectors
File Size      : 763k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

https://github.com/user-attachments/assets/88d4e88f-ebc4-4f32-b216-16d46b966024


To export the checkpoint to onnx:
```bash
python3 ./matcha/export_onnx.py \
  --exp-dir ./matcha/exp-1 \
  --epoch 2000 \
  --tokens ./data/tokens.txt \
  --cmvn ./data/fbank/cmvn.json
```

The above command generates the following files:
```
-rw-r--r-- 1 kuangfangjun root 72M Dec 27 18:53 model-steps-2.onnx
-rw-r--r-- 1 kuangfangjun root 73M Dec 27 18:54 model-steps-3.onnx
-rw-r--r-- 1 kuangfangjun root 73M Dec 27 18:54 model-steps-4.onnx
-rw-r--r-- 1 kuangfangjun root 74M Dec 27 18:55 model-steps-5.onnx
-rw-r--r-- 1 kuangfangjun root 74M Dec 27 18:57 model-steps-6.onnx
```

where the 2 in `model-steps-2.onnx` means it uses 2 steps for the ODE solver.

**HINT**: If you get the following error while running `export_onnx.py`:

```
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator
'aten::scaled_dot_product_attention' to ONNX opset version 14 is not supported.
```

please use `torch>=2.2.0`.

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

**HINT**: You can download pre-exported hifigan ONNX models from
<https://github.com/k2-fsa/sherpa-onnx/releases/tag/vocoder-models>

To use the generated onnx files to generate speech from text, please run:

```bash

# First, generate ./lexicon.txt
python3 ./matcha/generate_lexicon.py

python3 ./matcha/onnx_pretrained.py \
  --acoustic-model ./model-steps-4.onnx \
  --vocoder ./hifigan_v2.onnx \
  --tokens ./data/tokens.txt \
  --lexicon ./lexicon.txt \
  --input-text "在一个阳光明媚的夏天，小马、小羊和小狗它们一块儿在广阔的草地上，嬉戏玩耍，这时小猴来了，还带着它心爱的足球活蹦乱跳地跑前、跑后教小马、小羊、小狗踢足球。" \
  --output-wav ./1.wav
```

```bash
soxi ./1.wav

Input File     : './1.wav'
Channels       : 1
Sample Rate    : 22050
Precision      : 16-bit
Duration       : 00:00:16.37 = 360960 samples ~ 1227.76 CDDA sectors
File Size      : 722k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM
```

https://github.com/user-attachments/assets/578d04bb-fee8-47e5-9984-a868dcce610e

