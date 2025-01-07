## Results

### zipformer (zipformer + pruned-transducer w/ CR-CTC)

See <https://github.com/k2-fsa/icefall/pull/1766> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### large-scale model, number of model parameters: 148824074, i.e., 148.8 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-transducer-with-CR-CTC-20241019>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| greedy_search                        | 1.9        | 3.96       | --epoch 50 --avg 26 |
| modified_beam_search                 | 1.88       | 3.95       | --epoch 50 --avg 26 |

The training command using 2 80G-A100 GPUs is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# for non-streaming model training:
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large-cr-ctc-rnnt \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 1 \
  --use-attention-decoder 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --ctc-loss-scale 0.1 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.02 \
  --time-mask-ratio 2.5 \
  --full-libri 1 \
  --max-duration 1400 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search; do
  ./zipformer/decode.py \
    --epoch 50 \
    --avg 26 \
    --exp-dir zipformer/exp-large-cr-ctc-rnnt \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 1 \
    --use-attention-decoder 0 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --max-duration 300 \
    --decoding-method $m
done
```

### zipformer (zipformer + CR-CTC-AED)

See <https://github.com/k2-fsa/icefall/pull/1766> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### large-scale model, number of model parameters: 174319650, i.e., 174.3 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-aed-20241020>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| attention-decoder-rescoring-no-ngram | 1.96       | 4.08       | --epoch 50 --avg 20 |

The training command using 2 80G-A100 GPUs is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# for non-streaming model training:
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large-cr-ctc-aed \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 1 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --ctc-loss-scale 0.1 \
  --attention-decoder-loss-scale 0.9 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.02 \
  --time-mask-ratio 2.5 \
  --full-libri 1 \
  --max-duration 1200 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
./zipformer/ctc_decode.py \
  --epoch 50 \
  --avg 20 \
  --exp-dir zipformer/exp-large-cr-ctc-aed/ \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 1 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --max-duration 200 \
  --decoding-method attention-decoder-rescoring-no-ngram
done
```

### zipformer (zipformer + CR-CTC)

See <https://github.com/k2-fsa/icefall/pull/1766> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### small-scale model, number of model parameters: 22118279, i.e., 22.1 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-cr-ctc-20241018>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-greedy-decoding                  | 2.57       | 5.95       | --epoch 50 --avg 25 |
| ctc-prefix-beam-search               | 2.52       | 5.85       | --epoch 50 --avg 25 |

The training command using 2 32G-V100 GPUs is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# for non-streaming model training:
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-small/ \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 0 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --base-lr 0.04 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.2 \
  --time-mask-ratio 2.5 \
  --full-libri 1 \
  --max-duration 850 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-greedy-search ctc-prefix-beam-search; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 25 \
    --exp-dir zipformer/exp-small \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 0 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --max-duration 600 \
    --decoding-method $m
done
```

##### medium-scale model, number of model parameters: 64250603, i.e., 64.3 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-greedy-decoding                  | 2.12       | 4.62       | --epoch 50 --avg 24 |
| ctc-prefix-beam-search               | 2.1        | 4.61       | --epoch 50 --avg 24 |

The training command using 4 32G-V100 GPUs is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 0 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.2 \
  --time-mask-ratio 2.5 \
  --full-libri 1 \
  --max-duration 700 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-greedy-search ctc-prefix-beam-search; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 24 \
    --exp-dir zipformer/exp \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 0 \
    --max-duration 600 \
    --decoding-method $m
done
```

##### large-scale model, number of model parameters: 147010094, i.e., 147.0 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-cr-ctc-20241018>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-greedy-decoding                  | 2.03       | 4.37       | --epoch 50 --avg 26 |
| ctc-prefix-beam-search               | 2.02       | 4.35       | --epoch 50 --avg 26 |

The training command using 2 80G-A100 GPUs is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# For non-streaming model training:
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --use-cr-ctc 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.2 \
  --time-mask-ratio 2.5 \
  --full-libri 1 \
  --max-duration 1400 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-greedy-search ctc-prefix-beam-search; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 26 \
    --exp-dir zipformer/exp-large \
    --use-cr-ctc 1 \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 0 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --max-duration 600 \
    --decoding-method $m
done
```

### zipformer (zipformer + CTC/AED)

See <https://github.com/k2-fsa/icefall/pull/1389> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### small-scale model, number of model parameters: 46282107, i.e., 46.3 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-ctc-attention-decoder-2024-07-09>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-decoding                         | 3.04       | 7.04       | --epoch 50 --avg 30 |
| attention-decoder-rescoring-no-ngram | 2.45       | 6.08       | --epoch 50 --avg 30 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
# For non-streaming model training:
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-small \
  --full-libri 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 1 \
  --ctc-loss-scale 0.1 \
  --attention-decoder-loss-scale 0.9 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --base-lr 0.04 \
  --max-duration 1700 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-decoding attention-decoder-rescoring-no-ngram; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 30 \
    --exp-dir zipformer/exp-small \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 1 \
    --attention-decoder-loss-scale 0.9 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --max-duration 100 \
    --causal 0 \
    --num-paths 100 \
    --decoding-method $m
done
```

##### medium-scale model, number of model parameters: 89987295, i.e., 90.0 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-ctc-attention-decoder-2024-07-08>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-decoding                         | 2.46       | 5.57       | --epoch 50 --avg 22 |
| attention-decoder-rescoring-no-ngram | 2.23       | 4.98       | --epoch 50 --avg 22 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --full-libri 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 1 \
  --ctc-loss-scale 0.1 \
  --attention-decoder-loss-scale 0.9 \
  --max-duration 1200 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-decoding attention-decoder-rescoring-no-ngram; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 22 \
    --exp-dir zipformer/exp \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 1 \
    --attention-decoder-loss-scale 0.9 \
    --max-duration 100 \
    --causal 0 \
    --num-paths 100 \
    --decoding-method $m
done
```

##### large-scale model, number of model parameters: 174319650, i.e., 174.3 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-ctc-attention-decoder-2024-05-26>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method                      | test-clean | test-other | comment             |
|--------------------------------------|------------|------------|---------------------|
| ctc-decoding                         | 2.29       | 5.14       | --epoch 50 --avg 29 |
| attention-decoder-rescoring-no-ngram | 2.1        | 4.57       | --epoch 50 --avg 29 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# For non-streaming model training:
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --full-libri 1 \
  --use-ctc 1 \
  --use-transducer 0 \
  --use-attention-decoder 1 \
  --ctc-loss-scale 0.1 \
  --attention-decoder-loss-scale 0.9 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --max-duration 1200 \
  --master-port 12345
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-decoding attention-decoder-rescoring-no-ngram; do
  ./zipformer/ctc_decode.py \
    --epoch 50 \
    --avg 29 \
    --exp-dir zipformer/exp-large \
    --use-ctc 1 \
    --use-transducer 0 \
    --use-attention-decoder 1 \
    --attention-decoder-loss-scale 0.9 \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192 \
    --max-duration 100 \
    --causal 0 \
    --num-paths 100 \
    --decoding-method $m
done
```


### zipformer (zipformer + pruned stateless transducer + CTC)

See <https://github.com/k2-fsa/icefall/pull/1111> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### normal-scaled model, number of model parameters: 65805511, i.e., 65.81 M

The tensorboard log can be found at
<https://tensorboard.dev/experiment/Lo3Qlad7TP68ulM2K0ixgQ/>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-transducer-ctc-2023-06-13>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Results of the CTC head:

| decoding method         | test-clean | test-other | comment            |
|-------------------------|------------|------------|--------------------|
| ctc-decoding            | 2.40       | 5.66       | --epoch 40 --avg 16 |
| 1best                   | 2.46       | 5.11       | --epoch 40 --avg 16 |
| nbest                   | 2.46       | 5.11       | --epoch 40 --avg 16 |
| nbest-rescoring         | 2.37       | 4.93       | --epoch 40 --avg 16 |
| whole-lattice-rescoring | 2.37       | 4.88       | --epoch 40 --avg 16 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-ctc-rnnt \
  --causal 0 \
  --use-transducer 1 \
  --use-ctc 1 \
  --ctc-loss-scale 0.2 \
  --full-libri 1 \
  --max-duration 1000
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
  ./zipformer/ctc_decode.py \
      --epoch 40 \
      --avg 16 \
      --exp-dir zipformer/exp-ctc-rnnt \
      --use-transducer 1 \
      --use-ctc 1 \
      --max-duration 300 \
      --causal 0 \
      --num-paths 100 \
      --nbest-scale 1.0 \
      --hlg-scale 0.6 \
      --decoding-method $m
done
```

### zipformer (zipformer + pruned stateless transducer)

See <https://github.com/k2-fsa/icefall/pull/1058> for more details.

[zipformer](./zipformer)

#### Non-streaming

##### normal-scaled model, number of model parameters: 65549011, i.e., 65.55 M

The tensorboard log can be found at
<https://tensorboard.dev/experiment/R2DT9Ju4QiadC4e2ioKh5A/>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-2023-05-15>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 2.27       | 5.1        | --epoch 30 --avg 9 |
| modified_beam_search | 2.25       | 5.06       | --epoch 30 --avg 9 |
| fast_beam_search     | 2.25       | 5.04       | --epoch 30 --avg 9 |
| greedy_search        | 2.23       | 4.96       | --epoch 40 --avg 16 |
| modified_beam_search | 2.21       | 4.91       | --epoch 40 --avg 16 |
| fast_beam_search     | 2.24       | 4.93       | --epoch 40 --avg 16 |
| greedy_search        | 2.22       | 4.87       | --epoch 50 --avg 25 |
| modified_beam_search | 2.21       | 4.79       | --epoch 50 --avg 25 |
| fast_beam_search     | 2.21       | 4.82       | --epoch 50 --avg 25 |
| modified_beam_search_shallow_fusion | 2.01 | 4.37 | --epoch 40 --avg 16 --beam-size 12 --lm-scale 0.3 |
| modified_beam_search_LODR | 1.94 | 4.17 | --epoch 40 --avg 16 --beam-size 12 --lm-scale 0.52 --LODR-scale -0.26 |
| modified_beam_search_rescore | 2.04 | 4.39 | --epoch 40 --avg 16 --beam-size 12 |
| modified_beam_search_rescore_LODR | 2.01  | 4.33 | --epoch 40 --avg 16 --beam-size 12  |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 0 \
  --full-libri 1 \
  --max-duration 1000
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 50 \
    --avg 25 \
    --use-averaged-model 1 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method $m
done
```

To decode with external language models, please refer to the documentation [here](https://k2-fsa.github.io/icefall/decoding-with-langugage-models/index.html).

We also support training Zipformer with AMP+bf16 format (requires bf16 support). See [here](https://github.com/k2-fsa/icefall/pull/1700) for more details and pre-trained models. **The same command can be used for decoding and exporting the model.**

The amp+bf16 training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 0 \
  --use-bf16 1 \
  --exp-dir zipformer/exp_amp_bf16 \
  --causal 0 \
  --full-libri 1 \
  --max-duration 1000
```

##### small-scaled model, number of model parameters: 23285615, i.e., 23.3 M

The tensorboard log can be found at
<https://tensorboard.dev/experiment/M9C8cYPWSN2MVBYaBIX3EQ/>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-small-2023-05-16>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 2.64       | 6.14       | --epoch 30 --avg 8 |
| modified_beam_search | 2.6        | 6.01       | --epoch 30 --avg 8 |
| fast_beam_search     | 2.62       | 6.06       | --epoch 30 --avg 8 |
| greedy_search        | 2.49       | 5.91       | --epoch 40 --avg 13 |
| modified_beam_search | 2.46       | 5.83       | --epoch 40 --avg 13 |
| fast_beam_search     | 2.46       | 5.87       | --epoch 40 --avg 13 |
| greedy_search        | 2.46       | 5.86       | --epoch 50 --avg 23 |
| modified_beam_search | 2.42       | 5.73       | --epoch 50 --avg 23 |
| fast_beam_search     | 2.46       | 5.78       | --epoch 50 --avg 23 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-small \
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,768,768,768,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --base-lr 0.04 \
  --full-libri 1 \
  --max-duration 1500
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 50 \
    --avg 23 \
    --exp-dir zipformer/exp-small \
    --max-duration 600 \
    --causal 0 \
    --decoding-method $m \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,768,768,768,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192
done
```

##### large-scaled model, number of model parameters: 148439574, i.e., 148.4 M

The tensorboard log can be found at
<https://tensorboard.dev/experiment/C5ZPE5u1So2ZwhYLKW0FVg/>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-2023-05-16>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 2.12       | 4.91       | --epoch 30 --avg 9 |
| modified_beam_search | 2.11       | 4.9        | --epoch 30 --avg 9 |
| fast_beam_search     | 2.13       | 4.93       | --epoch 30 --avg 9 |
| greedy_search        | 2.12       | 4.8        | --epoch 40 --avg 13 |
| modified_beam_search | 2.11       | 4.7        | --epoch 40 --avg 13 |
| fast_beam_search     | 2.13       | 4.78       | --epoch 40 --avg 13 |
| greedy_search        | 2.08       | 4.69       | --epoch 50 --avg 30 |
| modified_beam_search | 2.06       | 4.63       | --epoch 50 --avg 30 |
| fast_beam_search     | 2.09       | 4.68       | --epoch 50 --avg 30 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --causal 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --full-libri 1 \
  --max-duration 1000
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 50 \
    --avg 30 \
    --exp-dir zipformer/exp-large \
    --max-duration 600 \
    --causal 0 \
    --decoding-method $m \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192
done
```

##### large-scaled model, number of model parameters: 148439574, i.e., 148.4 M, trained on 8 80G-A100 GPUs

The tensorboard log can be found at
<https://tensorboard.dev/experiment/95TdNyEuQXaWK2PzFpD9yg/>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-large-2023-10-26-8-a100>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | test-clean | test-other | comment               |
|----------------------|------------|------------|-----------------------|
| greedy_search        | 2.00       | 4.47       | --epoch 174 --avg 172 |
| modified_beam_search | 2.00       | 4.38       | --epoch 174 --avg 172 |
| fast_beam_search     | 2.00       | 4.42       | --epoch 174 --avg 172 |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 8 \
  --num-epochs 174 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --causal 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --full-libri 1 \
  --max-duration 2200
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 174 \
    --avg 172 \
    --exp-dir zipformer/exp-large \
    --max-duration 600 \
    --causal 0 \
    --decoding-method $m \
    --num-encoder-layers 2,2,4,5,4,2 \
    --feedforward-dim 512,768,1536,2048,1536,768 \
    --encoder-dim 192,256,512,768,512,256 \
    --encoder-unmasked-dim 192,192,256,320,256,192
done
```

#### streaming

##### normal-scaled model, number of model parameters: 66110931, i.e., 66.11 M

The tensorboard log can be found at
<https://tensorboard.dev/experiment/9rD0i6rMSWq1O61poWi71A>

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | chunk size | test-clean | test-other | decoding mode       | comment            |
|----------------------|------------|------------|------------|---------------------|--------------------|
| greedy_search        | 320ms      | 3.06       | 7.81       | simulated streaming | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| greedy_search        | 320ms      | 3.06       | 7.79       | chunk-wise          | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| modified_beam_search | 320ms      | 3.01       | 7.69       | simulated streaming | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| modified_beam_search | 320ms      | 3.05       | 7.69       | chunk-wise          | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| fast_beam_search     | 320ms      | 3.04       | 7.68       | simulated streaming | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| fast_beam_search     | 320ms      | 3.07       | 7.69       | chunk-wise          | --epoch 30 --avg 8 --chunk-size 16 --left-context-frames 128 |
| greedy_search        | 640ms      | 2.81       | 7.15       | simulated streaming | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |
| greedy_search        | 640ms      | 2.84       | 7.16       | chunk-wise          | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |
| modified_beam_search | 640ms      | 2.79       | 7.05       | simulated streaming | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |
| modified_beam_search | 640ms      | 2.81       | 7.11       | chunk-wise          | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |
| fast_beam_search     | 640ms      | 2.84       | 7.04       | simulated streaming | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |
| fast_beam_search     | 640ms      | 2.83       | 7.1        | chunk-wise          | --epoch 30 --avg 8 --chunk-size 32 --left-context-frames 256 |

Note: For decoding mode, `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
  while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-causal \
  --causal 1 \
  --full-libri 1 \
  --max-duration 1000
```

The simulated streaming decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 30 \
    --avg 8 \
    --use-averaged-model 1 \
    --exp-dir ./zipformer/exp-causal \
    --causal 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --max-duration 600 \
    --decoding-method $m
done
```

The chunk-wise streaming decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/streaming_decode.py \
    --epoch 30 \
    --avg 8 \
    --use-averaged-model 1 \
    --exp-dir ./zipformer/exp-causal \
    --causal 1 \
    --chunk-size 16 \
    --left-context-frames 128 \
    --num-decode-streams 2000 \
    --decoding-method $m
done
```

### Zipformer CTC

#### [zipformer_ctc](./zipformer_ctc)

See <https://github.com/k2-fsa/icefall/pull/941> for more details.

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/desh2608/icefall-asr-librispeech-zipformer-ctc>

Number of model parameters: 86083707, i.e., 86.08 M

| decoding method         | test-clean | test-other | comment             |
|-------------------------|------------|------------|---------------------|
| ctc-decoding            | 2.50       | 5.86       | --epoch 30 --avg 9  |
| whole-lattice-rescoring | 2.44       | 5.38       | --epoch 30 --avg 9  |
| attention-rescoring     | 2.35       | 5.16       | --epoch 30 --avg 9  |
| 1best                   | 2.01       | 4.61       | --epoch 30 --avg 9  |

The training commands are:
```bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer_ctc/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer_ctc/exp \
  --full-libri 1 \
  --max-duration 1000 \
  --master-port 12345
```

The tensorboard log can be found at:
<https://tensorboard.dev/experiment/IjPSJjHOQFKPYA5Z0Vf8wg>

The decoding command is:

```bash
./zipformer_ctc/decode.py \
  --epoch 30 --avg 9 --use-averaged-model True \
  --exp-dir zipformer_ctc/exp \
  --lang-dir data/lang_bpe_500 \
  --lm-dir data/lm \
  --method ctc-decoding
```

### pruned_transducer_stateless7 (Fine-tune with mux)

See <https://github.com/k2-fsa/icefall/pull/1059> for more details.

[pruned_transducer_stateless7](./pruned_transducer_stateless7)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/MaNDZfO7RzW2Czzf3R2ZRA/>

You can find the pretrained model and bpe model needed for fine-tuning at:
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11>

You can find a fine-tuned model, fine-tuning logs, decoding logs, and decoding
results at:
<https://huggingface.co/yfyeung/icefall-asr-finetune-mux-pruned_transducer_stateless7-2023-05-19>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 70369391, i.e., 70.37 M

| decoding method      |    dev     |    test    | test-clean | test-other |      comment       |
|----------------------|------------|------------|------------|------------|--------------------|
| greedy_search        |   14.27    |   14.22    |    2.08    |    4.79    | --epoch 20 --avg 5 |
| modified_beam_search |   14.22    |   14.08    |    2.06    |    4.72    | --epoch 20 --avg 5 |
| fast_beam_search     |   14.23    |   14.17    |    2.08    |    4.09    | --epoch 20 --avg 5 |

The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1"

./pruned_transducer_stateless7/finetune.py \
  --world-size 2 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless7/exp_giga_finetune \
  --subset S \
  --use-fp16 1 \
  --base-lr 0.005 \
  --lr-epochs 100 \
  --lr-batches 100000 \
  --bpe-model icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/data/lang_bpe_500/bpe.model \
  --do-finetune True \
  --use-mux True \
  --finetune-ckpt icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrain.pt \
  --max-duration 500
```

The decoding commands are:
```bash
# greedy_search
./pruned_transducer_stateless7/decode.py \
    --epoch 20 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp_giga_finetune \
    --max-duration 600 \
    --decoding-method greedy_search

# modified_beam_search
./pruned_transducer_stateless7/decode.py \
    --epoch 20 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp_giga_finetune \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

# fast_beam_search
./pruned_transducer_stateless7/decode.py \
    --epoch 20 \
    --avg 5 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp_giga_finetune \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
```

### pruned_transducer_stateless7 (zipformer + multidataset(LibriSpeech + GigaSpeech + CommonVoice 13.0))

See <https://github.com/k2-fsa/icefall/pull/1010> for more details.

[pruned_transducer_stateless7](./pruned_transducer_stateless7)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/SwdJoHgZSZWn8ph9aJLb8g/>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/yfyeung/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 70369391, i.e., 70.37 M

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 1.91       | 4.06       | --epoch 30 --avg 7 |
| modified_beam_search | 1.90       | 3.99       | --epoch 30 --avg 7 |
| fast_beam_search     | 1.90       | 3.98       | --epoch 30 --avg 7 |


The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless7/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --use-multidataset 1 \
  --use-fp16 1 \
  --max-duration 750 \
  --exp-dir pruned_transducer_stateless7/exp
```

The decoding commands are:
```bash
# greedy_search
./pruned_transducer_stateless7/decode.py \
    --epoch 30 \
    --avg 7 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method greedy_search

# modified_beam_search
./pruned_transducer_stateless7/decode.py \
    --epoch 30 \
    --avg 7 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --beam-size 4

# fast_beam_search
./pruned_transducer_stateless7/decode.py \
    --epoch 30 \
    --avg 7 \
    --use-averaged-model 1 \
    --exp-dir ./pruned_transducer_stateless7/exp \
    --max-duration 600 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
```

### Streaming Zipformer-Transducer (Pruned Stateless Transducer + Streaming Zipformer + Multi-Dataset)

#### [pruned_transducer_stateless7_streaming_multi](./pruned_transducer_stateless7_streaming_multi)

See <https://github.com/k2-fsa/icefall/pull/984> for more details.

You can find a pretrained model, training logs, decoding logs, and decoding
results at: <https://huggingface.co/marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04>

Number of model parameters: 70369391, i.e., 70.37 M

##### training on full librispeech + full gigaspeech (with giga_prob=0.9)

The WERs are:


| decoding method      | chunk size | test-clean | test-other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 2.43       | 6.0        | --epoch 20 --avg 4  | simulated streaming  |
| greedy search        | 320ms      | 2.47       | 6.13       | --epoch 20 --avg 4  | chunk-wise           |
| fast beam search     | 320ms      | 2.43       | 5.99       | --epoch 20 --avg 4  | simulated streaming  |
| fast beam search     | 320ms      | 2.8        | 6.46       | --epoch 20 --avg 4  | chunk-wise           |
| modified beam search | 320ms      | 2.4        | 5.96       | --epoch 20 --avg 4  | simulated streaming  |
| modified beam search | 320ms      | 2.42       | 6.03       | --epoch 20 --avg 4  | chunk-size           |
| greedy search        | 640ms      | 2.26       | 5.58       | --epoch 20 --avg 4  | simulated streaming  |
| greedy search        | 640ms      | 2.33       | 5.76       | --epoch 20 --avg 4  | chunk-wise           |
| fast beam search     | 640ms      | 2.27       | 5.54       | --epoch 20 --avg 4  | simulated streaming  |
| fast beam search     | 640ms      | 2.37       | 5.75       | --epoch 20 --avg 4  | chunk-wise           |
| modified beam search | 640ms      | 2.22       | 5.5        | --epoch 20 --avg 4  | simulated streaming  |
| modified beam search | 640ms      | 2.25       | 5.69       | --epoch 20 --avg 4  | chunk-size           |

The model also has good WERs on GigaSpeech. The following WERs are achieved on GigaSpeech test and dev sets:

| decoding method      | chunk size | dev        | test | comment    | decoding mode       |
|----------------------|------------|-----|------|------------|---------------------|
| greedy search        | 320ms      | 12.08       | 11.98       | --epoch 20 --avg 4  | simulated streaming  |
| greedy search        | 640ms      | 11.66       | 11.71       | --epoch 20 --avg 4  | simulated streaming  |
| modified beam search | 320ms      | 11.95       | 11.83       | --epoch 20 --avg 4  | simulated streaming  |
| modified beam search | 320ms      | 11.65       | 11.56       | --epoch 20 --avg 4  | simulated streaming  |


Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command is:

```bash
./pruned_transducer_stateless7_streaming_multi/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp \
  --full-libri 1 \
  --giga-prob 0.9 \
  --max-duration 750 \
  --master-port 12345
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/G4yDMLXGQXexf41i4MA2Tg/#scalars>

The simulated streaming decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  ./pruned_transducer_stateless7_streaming_multi/decode.py \
    --epoch 20 \
    --avg 4 \
    --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --right-padding 64 \
    --decoding-method $m
done
```

The streaming chunk-size decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless7_streaming_multi/streaming_decode.py \
    --epoch 20 \
    --avg 4 \
    --exp-dir ./pruned_transducer_stateless7_streaming_multi/exp \
    --decoding-method $m \
    --decode-chunk-len 32 \
    --num-decode-streams 2000
done
```

#### Smaller model

We also provide a very small version (only 6.1M parameters) of this setup. The training command for the small model is:

```bash
./pruned_transducer_stateless7_streaming_multi/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming_multi/exp \
  --full-libri 1 \
  --giga-prob 0.9 \
  --num-encoder-layers "2,2,2,2,2" \
  --feedforward-dims "256,256,512,512,256" \
  --nhead "4,4,4,4,4" \
  --encoder-dims "128,128,128,128,128" \
  --attention-dims "96,96,96,96,96" \
  --encoder-unmasked-dims "96,96,96,96,96" \
  --max-duration 1200 \
  --master-port 12345
```

You can find this pretrained small model and its training logs, decoding logs, and decoding
results at:
<https://huggingface.co/marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-6M-2023-04-03>


| decoding method      | chunk size | test-clean | test-other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 5.95       | 15.03       | --epoch 30 --avg 1  | simulated streaming  |
| greedy search        | 640ms      | 5.61       | 13.86       | --epoch 30 --avg 1  | simulated streaming  |
| modified beam search | 320ms      | 5.72       | 14.34      | --epoch 30 --avg 1  | simulated streaming  |
| modified beam search | 640ms      | 5.43       | 13.16      | --epoch 30 --avg 1  | simulated streaming  |
| fast beam search | 320ms      | 5.88       | 14.45      | --epoch 30 --avg 1  | simulated streaming  |
| fast beam search | 640ms      | 5.48       | 13.31      | --epoch 30 --avg 1  | simulated streaming  |

This small model achieves the following WERs on GigaSpeech test and dev sets:

| decoding method      | chunk size | dev | test | comment    | decoding mode       |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 17.57       | 17.2     | --epoch 30 --avg 1  | simulated streaming  |
| modified beam search | 320ms      | 16.98       | 11.98       | --epoch 30 --avg 1  | simulated streaming  |

You can find the tensorboard logs at <https://tensorboard.dev/experiment/tAc5iXxTQrCQxky5O5OLyw/#scalars>.


### Streaming Zipformer-Transducer (Pruned Stateless Transducer + Streaming Zipformer)

#### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

See <https://github.com/k2-fsa/icefall/pull/787> for more details.

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29>

Number of model parameters: 70369391, i.e., 70.37 M

##### training on full librispeech

The WERs are:

| decoding method      | chunk size | test-clean | test-other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 3.15       | 8.09       | --epoch 30 --avg 9  | simulated streaming  |
| greedy search        | 320ms      | 3.17       | 8.24       | --epoch 30 --avg 9  | chunk-wise           |
| fast beam search     | 320ms      | 3.2        | 8.04       | --epoch 30 --avg 9  | simulated streaming  |
| fast beam search     | 320ms      | 3.36       | 8.19       | --epoch 30 --avg 9  | chunk-wise           |
| modified beam search | 320ms      | 3.11       | 7.93       | --epoch 30 --avg 9  | simulated streaming  |
| modified beam search | 320ms      | 3.12       | 8.11       | --epoch 30 --avg 9  | chunk-size           |
| greedy search        | 640ms      | 2.97       | 7.5        | --epoch 30 --avg 9  | simulated streaming  |
| greedy search        | 640ms      | 2.98       | 7.67       | --epoch 30 --avg 9  | chunk-wise           |
| fast beam search     | 640ms      | 3.02       | 7.47       | --epoch 30 --avg 9  | simulated streaming  |
| fast beam search     | 640ms      | 2.96       | 7.61       | --epoch 30 --avg 9  | chunk-wise           |
| modified beam search | 640ms      | 2.94       | 7.36       | --epoch 30 --avg 9  | simulated streaming  |
| modified beam search | 640ms      | 2.95       | 7.53       | --epoch 30 --avg 9  | chunk-size           |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command is:

```bash
./pruned_transducer_stateless7_streaming/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --full-libri 1 \
  --max-duration 750 \
  --master-port 12345
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/A46UpqEWQWS7oDi5VcQ8rg/>

The simulated streaming decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  ./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 30 \
    --avg 9 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method $m
done
```

The streaming chunk-size decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless7_streaming/streaming_decode.py \
    --epoch 30 \
    --avg 9 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --decoding-method $m \
    --decode-chunk-len 32 \
    --num-decode-streams 2000
done
```
We also support decoding with neural network LMs. After combining with language models, the WERs are
| decoding method      | chunk size | test-clean | test-other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| `modified_beam_search` | 320ms      | 3.11       | 7.93       | --epoch 30 --avg 9  | simulated streaming  |
| `modified_beam_search_lm_shallow_fusion` | 320ms      | 2.58       | 6.65       | --epoch 30 --avg 9  | simulated streaming  |
| `modified_beam_search_lm_rescore` | 320ms      | 2.59       | 6.86       | --epoch 30 --avg 9  | simulated streaming  |
| `modified_beam_search_lm_rescore_LODR` | 320ms      | 2.52       | 6.73       | --epoch 30 --avg 9  | simulated streaming  |

Please use the following command for `modified_beam_search_lm_shallow_fusion`:
```bash
for lm_scale in $(seq 0.15 0.01 0.38); do
    for beam_size in 4 8 12; do
        ./pruned_transducer_stateless7_streaming/decode.py \
            --epoch 99 \
            --avg 1 \
            --use-averaged-model False \
            --beam-size $beam_size \
            --exp-dir ./pruned_transducer_stateless7_streaming/exp-large-LM \
            --max-duration 600 \
            --decode-chunk-len 32 \
            --decoding-method modified_beam_search_lm_shallow_fusion \
            --use-shallow-fusion 1 \
            --lm-type rnn \
            --lm-exp-dir rnn_lm/exp \
            --lm-epoch 99 \
            --lm-scale $lm_scale \
            --lm-avg 1 \
            --rnn-lm-embedding-dim 2048 \
            --rnn-lm-hidden-dim 2048 \
            --rnn-lm-num-layers 3 \
            --lm-vocab-size 500
    done
done
```

Please use the following command for `modified_beam_search_lm_rescore`:
```bash
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model True \
    --beam-size 8 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method modified_beam_search_lm_rescore \
    --use-shallow-fusion 0 \
    --lm-type rnn \
    --lm-exp-dir rnn_lm/exp \
    --lm-epoch 99 \
    --lm-avg 1 \
    --rnn-lm-embedding-dim 2048 \
    --rnn-lm-hidden-dim 2048 \
    --rnn-lm-num-layers 3 \
    --lm-vocab-size 500
```

Please use the following command for `modified_beam_search_lm_rescore_LODR`:
```bash
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 30 \
    --avg 9 \
    --use-averaged-model True \
    --beam-size 8 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method modified_beam_search_lm_rescore_LODR \
    --use-shallow-fusion 0 \
    --lm-type rnn \
    --lm-exp-dir rnn_lm/exp \
    --lm-epoch 99 \
    --lm-avg 1 \
    --rnn-lm-embedding-dim 2048 \
    --rnn-lm-hidden-dim 2048 \
    --rnn-lm-num-layers 3 \
    --lm-vocab-size 500 \
    --tokens-ngram 2 \
    --backoff-id 500
```

A well-trained RNNLM can be found here: <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>. The bi-gram used in LODR decoding
can be found here: <https://huggingface.co/marcoyang/librispeech_bigram>.


#### Smaller model

A smaller model (~20M params) is also available with configuration based on [this comment](https://github.com/k2-fsa/icefall/pull/745#issuecomment-1405282740). The WERs are:

| decoding method      | chunk size | test-clean | test-other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 3.94       | 9.79       | --epoch 30 --avg 9  | simulated streaming  |
| modified beam search | 320ms      | 3.88       | 9.53       | --epoch 30 --avg 9  | simulated streaming  |

You can find a pretrained model, training logs, decoding logs, and decoding
results at: <https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small>


### zipformer_mmi (zipformer with mmi loss)

See <https://github.com/k2-fsa/icefall/pull/746> for more details.

[zipformer_mmi](./zipformer_mmi)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/xyOZUKpEQm62HBIlUD4uPA/>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-mmi-2022-12-08>

Number of model parameters: 69136519, i.e., 69.14 M

|                        | test-clean | test-other | comment             |
| ---------------------- | ---------- | ---------- | ------------------- |
| 1best                  | 2.54       | 5.65       | --epoch 30 --avg 10 |
| nbest                  | 2.54       | 5.66       | --epoch 30 --avg 10 |
| nbest-rescoring-LG     | 2.49       | 5.42       | --epoch 30 --avg 10 |
| nbest-rescoring-3-gram | 2.52       | 5.62       | --epoch 30 --avg 10 |
| nbest-rescoring-4-gram | 2.5        | 5.51       | --epoch 30 --avg 10 |

The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer_mmi/train.py \
  --world-size 4 \
  --master-port 12345 \
  --num-epochs 30 \
  --start-epoch 1 \
  --lang-dir data/lang_bpe_500 \
  --max-duration 500 \
  --full-libri 1 \
  --use-fp16 1 \
  --exp-dir zipformer_mmi/exp
```

The decoding commands for the transducer branch are:
```bash
export CUDA_VISIBLE_DEVICES="5"

for m in nbest nbest-rescoring-LG nbest-rescoring-3-gram nbest-rescoring-4-gram; do
  ./zipformer_mmi/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./zipformer_mmi/exp/ \
    --max-duration 100 \
    --lang-dir data/lang_bpe_500 \
    --nbest-scale 1.2 \
    --hp-scale 1.0 \
    --decoding-method $m
done
```

### pruned_transducer_stateless7_ctc_bs (zipformer with transducer loss and ctc loss using blank skip)

See https://github.com/k2-fsa/icefall/pull/730 for more details.

[pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/rrNZ7l83Qu6RKoD7y49wiA/>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29>

Number of model parameters: 76804822, i.e., 76.80 M

Test on 8-card V100 cluster, with 4-card busy and 4-card idle.

#### greedy_search

| model                                                        | test-clean | test-other | decoding time(s) | comment             |
| ------------------------------------------------------------ | ---------- | ---------- | ---------------- | ------------------- |
| [pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs) | 2.28       | 5.53       | 48.939           | --epoch 30 --avg 13 |
| [pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc) | 2.24       | 5.18       | 91.900           | --epoch 30 --avg 8  |

- [pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs) applies blank skip both on training and decoding, and [pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc) doesn`t apply blank skip.
- Applying blank skip both on training and decoding is **1.88 times** faster than the model that doesn't apply blank skip without obvious performance loss.

#### modified_beam_search

| model                                                        | test-clean | test-other | decoding time(s) | comment             |
| ------------------------------------------------------------ | ---------- | ---------- | ---------------- | ------------------- |
| [pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs) | 2.26       | 5.44       | 80.446           | --epoch 30 --avg 13 |
| [pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc) | 2.20       | 5.12       | 283.676          | --epoch 30 --avg 8  |

- [pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs) applies blank skip both on training and decoding, and [pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc) doesn`t apply blank skip.
- Applying blank skip both on training and decoding is **3.53 times**  faster than the model that doesn't apply blank skip without obvious performance loss.

The training commands for the model using blank skip ([pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs)) are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_ctc_bs/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 750 \
  --exp-dir pruned_transducer_stateless7_ctc_bs/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --ctc-loss-scale 0.2 \
  --master-port 12535
```

The decoding commands for the transducer branch of the model using blank skip ([pruned_transducer_stateless7_ctc_bs](./pruned_transducer_stateless7_ctc_bs)) are:

```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  for epoch in 30; do
    for avg in 15; do
      ./pruned_transducer_stateless7_ctc_bs/ctc_guide_decode_bs.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```

The decoding commands for the transducer branch of the model without blank skip ([pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc)) are:

```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  for epoch in 30; do
    for avg in 15; do
      ./pruned_transducer_stateless7_ctc/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./pruned_transducer_stateless7_ctc/exp \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```

### pruned_transducer_stateless7_ctc (zipformer with transducer loss and ctc loss)

See <https://github.com/k2-fsa/icefall/pull/683> for more details.

[pruned_transducer_stateless7_ctc](./pruned_transducer_stateless7_ctc)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/hxlGAhOPToGmRLZFnAzPWw/>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01>

Number of model parameters: 70561891, i.e., 70.56 M

|                          | test-clean | test-other  | comment            |
|--------------------------|------------|-------------|--------------------|
| greedy search            | 2.23       | 5.19        | --epoch 30 --avg 8 |
| modified beam search     | 2.21       | 5.12        | --epoch 30 --avg 8 |
| fast beam search         | 2.23       | 5.18        | --epoch 30 --avg 8 |
| ctc decoding             | 2.48       | 5.82        | --epoch 30 --avg 9 |
| 1best                    | 2.43       | 5.22        | --epoch 30 --avg 9 |
| nbest                    | 2.43       | 5.22        | --epoch 30 --avg 9 |
| nbest rescoring          | 2.34       | 5.05        | --epoch 30 --avg 9 |
| whole lattice rescoring  | 2.34       | 5.04        | --epoch 30 --avg 9 |

The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_ctc/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 750 \
  --exp-dir pruned_transducer_stateless7_ctc/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --ctc-loss-scale 0.2 \
  --master-port 12535
```

The decoding commands for the transducer branch are:
```bash
for m in greedy_search fast_beam_search modified_beam_search ; do
  for epoch in 30; do
    for avg in 8; do
      ./pruned_transducer_stateless7_ctc/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./pruned_transducer_stateless7_ctc/exp \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```

The decoding commands for the ctc branch are:
```bash
for m in ctc-decoding nbest nbest-rescoring whole-lattice-rescoring; do
  for epoch in 30; do
    for avg in 9; do
      ./pruned_transducer_stateless7_ctc/ctc_decode.py \
          --epoch $epoch \
          --avg $avg \
          --exp-dir ./pruned_transducer_stateless7_ctc/exp \
          --max-duration 100 \
          --decoding-method $m \
          --hlg-scale 0.6 \
          --lm-dir data/lm
    done
  done
done
```


### LibriSpeech BPE training results (Conformer CTC, supporting delay penalty)

#### [conformer_ctc3](./conformer_ctc3)

It implements Conformer model training with CTC loss.
For streaming mode, it supports symbol delay penalty.

See <https://github.com/k2-fsa/icefall/pull/669> for more details.

##### training on full librispeech

This model contains 12 encoder layers. The number of model parameters is 77352694.

The WERs are:

|                                     | test-clean | test-other | comment              |
|-------------------------------------|------------|------------|----------------------|
| ctc-decoding                        | 3.09       | 7.62       | --epoch 25 --avg 7   |
| 1best                               | 2.87       | 6.44       | --epoch 25 --avg 7   |
| nbest                               | 2.88       | 6.5        | --epoch 25 --avg 7   |
| nbest-rescoring                     | 2.71       | 6.1        | --epoch 25 --avg 7   |
| whole-lattice-rescoring             | 2.71       | 6.04       | --epoch 25 --avg 7   |

The training command is:

```bash
./conformer_ctc3/train.py \
  --world-size 4 \
  --num-epochs 25 \
  --start-epoch 1 \
  --exp-dir conformer_ctc3/full \
  --full-libri 1 \
  --max-duration 300 \
  --master-port 12345
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/4jbxIQ2SQIaQeRqsR6bOSA>

The decoding command using different methods is:
```bash
for method in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
  ./conformer_ctc3/decode.py \
    --epoch 25 \
    --avg 7 \
    --exp-dir conformer_ctc3/exp \
    --max-duration 300 \
    --decoding-method $method \
    --manifest-dir data/fbank \
    --lm-dir data/lm \
done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-conformer-ctc3-2022-11-27>

The command to train a streaming model with symbol delay penalty is:
```bash
./conformer_ctc3/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir conformer_ctc3/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 25 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --delay-penalty 0.1
```
To evaluate symbol delay, you should:
(1) Generate cuts with word-time alignments:
```bash
./local/add_alignment_librispeech.py \
  --alignments-dir data/alignment \
  --cuts-in-dir data/fbank \
  --cuts-out-dir data/fbank_ali
```
(2) Set the argument "--manifest-dir data/fbank_ali" while decoding.
For example:
```bash
./conformer_ctc3/decode.py \
  --epoch 25 \
  --avg 7 \
  --exp-dir ./conformer_ctc3/exp \
  --max-duration 300 \
  --decoding-method ctc-decoding \
  --simulate-streaming 1 \
  --causal-convolution 1 \
  --decode-chunk-size 16 \
  --left-context 64 \
  --manifest-dir data/fbank_ali
```
Note: It supports to calculate symbol delay with following decoding methods:
  - ctc-greedy-search
  - ctc-decoding
  - 1best


### pruned_transducer_stateless8 (zipformer + multidataset)

See <https://github.com/k2-fsa/icefall/pull/675> for more details.

[pruned_transducer_stateless8](./pruned_transducer_stateless8)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/3e9AfOcgRwOXpLQlZvHZrQ>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 70369391, i.e., 70.37 M

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 1.81       | 4.18       | --epoch 20 --avg 4 |
| fast_beam_search     | 1.82       | 4.15       | --epoch 20 --avg 4 |
| modified_beam_search | 1.78       | **4.08**   | --epoch 20 --avg 4 |
| greedy_search        | 1.84       | 4.3        | --epoch 19 --avg 8 |
| fast_beam_search     |**1.77**    | 4.25       | --epoch 19 --avg 8 |
| modified_beam_search | 1.81       | 4.16       | --epoch 19 --avg 8 |


The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless8/train.py \
  --world-size 8 \
  --num-epochs 20 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 750 \
  --exp-dir pruned_transducer_stateless8/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535 \
  --giga-prob 0.9
```

The decoding commands are:
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  for epoch in $(seq 20 -1 10); do
    for avg in $(seq 9 -1 1); do
      ./pruned_transducer_stateless8/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./pruned_transducer_stateless8/exp \
          --feedforward-dims "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```


### pruned_transducer_stateless7 (zipformer)

See <https://github.com/k2-fsa/icefall/pull/672> for more details.

[pruned_transducer_stateless7](./pruned_transducer_stateless7)

The tensorboard log can be found at
<https://tensorboard.dev/experiment/P7vXWqK7QVu1mU9Ene1gGg/>

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 70369391, i.e., 70.37 M

|                      | test-clean | test-other  | comment                                |
|----------------------|------------|-------------|----------------------------------------|
| greedy search        | 2.17       | 5.23        | --epoch 30 --avg 9 --max-duration 600  |
| modified beam search | 2.15       | 5.20        | --epoch 30 --avg 9 --max-duration 600  |
| modified beam search + RNNLM shallow fusion | 1.99       | 4.73        | --epoch 30 --avg 9 --max-duration 600  |
| modified beam search + TransformerLM shallow fusion | 1.94       | 4.73        | --epoch 30 --avg 9 --max-duration 600  |
| modified beam search + RNNLM + LODR | 1.91       | 4.57        | --epoch 30 --avg 9 --max-duration 600  |
| modified beam search + TransformerLM + LODR | 1.91       | 4.51        | --epoch 30 --avg 9 --max-duration 600  |
| fast beam search     | 2.15       | 5.22        | --epoch 30 --avg 9 --max-duration 600  |

The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,3,6,7"

./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --full-libri 1 \
  --use-fp16 1 \
  --max-duration 750 \
  --exp-dir pruned_transducer_stateless7/exp \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535
```

The decoding commands are:
```bash
for m in greedy_search fast_beam_search modified_beam_search ; do
  for epoch in 30; do
    for avg in 9; do
      ./pruned_transducer_stateless7/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model 1 \
          --exp-dir ./pruned_transducer_stateless7/exp \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```

Note that a small change is made to the `pruned_transducer_stateless7/decoder.py` in
this [PR](https://github.com/k2-fsa/icefall/pull/942) to address the
problem of emitting the first symbol at the very beginning. If you need a
model without this issue, please download the model from here: <https://huggingface.co/marcoyang/icefall-asr-librispeech-pruned-transducer-stateless7-2023-03-10>

### LibriSpeech BPE training results (Pruned Stateless LSTM RNN-T + gradient filter)

#### [lstm_transducer_stateless3](./lstm_transducer_stateless3)

It implements LSTM model with mechanisms in reworked model for streaming ASR.
Gradient filter is applied inside each lstm module to stabilize the training.

See <https://github.com/k2-fsa/icefall/pull/564> for more details.

##### training on full librispeech

This model contains 12 encoder layers (LSTM module + Feedforward module). The number of model parameters is 84689496.

The WERs are:

|                                     | test-clean | test-other | comment              | decoding mode        |
|-------------------------------------|------------|------------|----------------------|----------------------|
| greedy search (max sym per frame 1) | 3.66       | 9.51       | --epoch 40 --avg 15  | simulated streaming  |
| greedy search (max sym per frame 1) | 3.66       | 9.48       | --epoch 40 --avg 15  | streaming            |
| fast beam search                    | 3.55       | 9.33       | --epoch 40 --avg 15  | simulated streaming  |
| fast beam search                    | 3.57       | 9.25       | --epoch 40 --avg 15  | streaming            |
| modified beam search                | 3.55       | 9.28       | --epoch 40 --avg 15  | simulated streaming  |
| modified beam search                | 3.54       | 9.25       | --epoch 40 --avg 15  | streaming            |

Note: `simulated streaming` indicates feeding full utterance during decoding, while `streaming` indicates feeding certain number of frames at each time.


The training command is:

```bash
./lstm_transducer_stateless3/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 1 \
  --exp-dir lstm_transducer_stateless3/exp \
  --full-libri 1 \
  --max-duration 500 \
  --master-port 12325 \
  --num-encoder-layers 12 \
  --grad-norm-threshold 25.0 \
  --rnn-hidden-size 1024
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/caNPyr5lT8qAl9qKsXEeEQ/>

The simulated streaming decoding command using greedy search, fast beam search, and modified beam search is:
```bash
for decoding_method in greedy_search fast_beam_search modified_beam_search; do
  ./lstm_transducer_stateless3/decode.py \
    --epoch 40 \
    --avg 15 \
    --exp-dir lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --use-averaged-model True \
    --beam 4 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 4
done
```

The streaming decoding command using greedy search, fast beam search, and modified beam search is:
```bash
for decoding_method in greedy_search fast_beam_search modified_beam_search; do
  ./lstm_transducer_stateless3/streaming_decode.py \
    --epoch 40 \
    --avg 15 \
    --exp-dir lstm_transducer_stateless3/exp \
    --max-duration 600 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --use-averaged-model True \
    --beam 4 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 4
done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless3-2022-09-28>


### LibriSpeech BPE training results (Pruned Stateless LSTM RNN-T + multi-dataset)

#### [lstm_transducer_stateless2](./lstm_transducer_stateless2)

See <https://github.com/k2-fsa/icefall/pull/558> for more details.

The WERs are:

|                                     | test-clean | test-other | comment                 |
|-------------------------------------|------------|------------|-------------------------|
| greedy search (max sym per frame 1) | 2.78       | 7.36       | --iter 468000 --avg 16  |
| modified_beam_search                | 2.73       | 7.15       | --iter 468000 --avg 16  |
| modified_beam_search + RNNLM shallow fusion   | 2.42     |  6.46      | --iter 468000 --avg 16  |
| modified_beam_search + TransformerLM shallow fusion   | 2.37     |  6.48      | --iter 468000 --avg 16  |
| modified_beam_search + RNNLM + LODR   | 2.24     |  5.89      | --iter 468000 --avg 16  |
| modified_beam_search + TransformerLM + LODR   | 2.19     |  5.90      | --iter 468000 --avg 16  |
| fast_beam_search                    | 2.76       | 7.31       | --iter 468000 --avg 16  |
| greedy search (max sym per frame 1) | 2.77       | 7.35       | --iter 472000 --avg 18  |
| modified_beam_search                | 2.75       | 7.08       | --iter 472000 --avg 18  |
| fast_beam_search                    | 2.77       | 7.29       | --iter 472000 --avg 18  |


The training command is:

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./lstm_transducer_stateless2/train.py \
  --world-size 8 \
  --num-epochs 35 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir lstm_transducer_stateless2/exp \
  --max-duration 500 \
  --use-fp16 0 \
  --lr-epochs 10 \
  --num-workers 2 \
  --giga-prob 0.9
```
**Note**: It was killed manually after getting `epoch-18.pt`. Also, we resumed
training after getting `epoch-9.pt`.

The tensorboard log can be found at
<https://tensorboard.dev/experiment/1ziQ2LFmQY2mt4dlUr5dyA/>

The decoding command is
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  for iter in 472000; do
    for avg in 8 10 12 14 16 18; do
      ./lstm_transducer_stateless2/decode.py \
        --iter $iter \
        --avg $avg \
        --exp-dir lstm_transducer_stateless2/exp \
        --max-duration 600 \
        --num-encoder-layers 12 \
        --rnn-hidden-size 1024 \
        --decoding-method $m \
        --use-averaged-model True \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8 \
        --beam-size 4
    done
  done
done
```

You may also decode using shallow fusion with external neural network LM. To do so you need to
download a well-trained NN LM:
RNN LM: <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>
Transformer LM: <https://huggingface.co/marcoyang/icefall-librispeech-transformer-lm/tree/main>

```bash
for iter in 472000; do
    for avg in 8 10 12 14 16 18; do
        ./lstm_transducer_stateless2/decode.py \
                --iter $iter \
                --avg $avg \
                --exp-dir ./lstm_transducer_stateless2/exp \
                --max-duration 600 \
                --decoding-method modified_beam_search_lm_shallow_fusion \
                --use-shallow-fusion 1 \
                --lm-type rnn \
                --lm-exp-dir /ceph-data4/yangxiaoyu/pretrained_models/LM/icefall-librispeech-rnn-lm/exp \
                --lm-epoch 99 \
                --lm-scale $lm_scale \
                --lm-avg 1 \
    done
done
```

You may also decode using LODR + LM shallow fusion. This decoding method is proposed in <https://arxiv.org/pdf/2203.16776.pdf>.
It subtracts the internal language model score during shallow fusion, which is approximated by a bi-gram model. The bi-gram can be
generated by `prepare_lm.sh` at stage 4, or you may download it from <https://huggingface.co/marcoyang/librispeech_bigram>.

The decoding command is as follows:

```bash
for iter in 472000; do
    for avg in 8 10 12 14 16 18; do
        ./lstm_transducer_stateless2/decode.py \
                --iter $iter \
                --avg $avg \
                --exp-dir ./lstm_transducer_stateless2/exp \
                --max-duration 600 \
                --decoding-method modified_beam_search_LODR \
                --beam 4 \
                --max-contexts 4 \
                --use-shallow-fusion 1 \
                --lm-type rnn \
                --lm-exp-dir /ceph-data4/yangxiaoyu/pretrained_models/LM/icefall-librispeech-rnn-lm/exp \
                --lm-epoch 99 \
                --lm-scale 0.4 \
                --lm-avg 1 \
                --tokens-ngram 2 \
                --ngram-lm-scale -0.16
    done
done
```
Note that you can also set `--lm-type transformer` to use transformer LM during LODR. But it will be slower
because it has not been optimized. The pre-trained transformer LM is available at <https://huggingface.co/marcoyang/icefall-librispeech-transformer-lm/tree/main>

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-lstm-transducer-stateless2-2022-09-03>


### LibriSpeech BPE training results (Pruned Stateless LSTM RNN-T)

#### [lstm_transducer_stateless](./lstm_transducer_stateless)

It implements LSTM model with mechanisms in reworked model for streaming ASR.

See <https://github.com/k2-fsa/icefall/pull/479> for more details.

##### training on full librispeech

This model contains 12 encoder layers (LSTM module + Feedforward module). The number of model parameters is 84689496.

The WERs are:

|                                     | test-clean | test-other | comment              | decoding mode        |
|-------------------------------------|------------|------------|----------------------|----------------------|
| greedy search (max sym per frame 1) | 3.81       | 9.73       | --epoch 35 --avg 15  | simulated streaming  |
| greedy search (max sym per frame 1) | 3.78       | 9.79       | --epoch 35 --avg 15  | streaming            |
| fast beam search                    | 3.74       | 9.59       | --epoch 35 --avg 15  | simulated streaming  |
| fast beam search                    | 3.73       | 9.61       | --epoch 35 --avg 15  | streaming            |
| modified beam search                | 3.64       | 9.55       | --epoch 35 --avg 15  | simulated streaming  |
| modified beam search                | 3.65       | 9.51       | --epoch 35 --avg 15  | streaming            |

Note: `simulated streaming` indicates feeding full utterance during decoding, while `streaming` indicates feeding certain number of frames at each time.

The training command is:

```bash
./lstm_transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 35 \
  --start-epoch 1 \
  --exp-dir lstm_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 500 \
  --master-port 12321 \
  --num-encoder-layers 12 \
  --rnn-hidden-size 1024
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/FWrM20mjTeWo6dTpFYOsYQ/>

The simulated streaming decoding command using greedy search, fast beam search, and modified beam search is:
```bash
for decoding_method in greedy_search fast_beam_search modified_beam_search; do
  ./lstm_transducer_stateless/decode.py \
    --epoch 35 \
    --avg 15 \
    --exp-dir lstm_transducer_stateless/exp \
    --max-duration 600 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --use-averaged-model True \
    --beam 4 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 4
done
```

The streaming decoding command using greedy search, fast beam search, and modified beam search is:
```bash
for decoding_method in greedy_search fast_beam_search modified_beam_search; do
  ./lstm_transducer_stateless/streaming_decode.py \
    --epoch 35 \
    --avg 15 \
    --exp-dir lstm_transducer_stateless/exp \
    --max-duration 600 \
    --num-encoder-layers 12 \
    --rnn-hidden-size 1024 \
    --decoding-method $decoding_method \
    --use-averaged-model True \
    --beam 4 \
    --max-contexts 4 \
    --max-states 8 \
    --beam-size 4
done
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-lstm-transducer-stateless-2022-08-18>


#### LibriSpeech BPE training results (Pruned Stateless Conv-Emformer RNN-T 2)

[conv_emformer_transducer_stateless2](./conv_emformer_transducer_stateless2)

It implements [Emformer](https://arxiv.org/abs/2010.10759) augmented with convolution module and simplified memory bank for streaming ASR.
It is modified from [torchaudio](https://github.com/pytorch/audio).

See <https://github.com/k2-fsa/icefall/pull/440> for more details.

##### With lower latency setup, training on full librispeech

In this model, the lengths of chunk and right context are 32 frames (i.e., 0.32s) and 8 frames (i.e., 0.08s), respectively.

The WERs are:

|                                     | test-clean | test-other | comment              | decoding mode        |
|-------------------------------------|------------|------------|----------------------|----------------------|
| greedy search (max sym per frame 1) | 3.5        | 9.09       | --epoch 30 --avg 10  | simulated streaming  |
| greedy search (max sym per frame 1) | 3.57       | 9.1        | --epoch 30 --avg 10  | streaming            |
| fast beam search                    | 3.5        | 8.91       | --epoch 30 --avg 10  | simulated streaming  |
| fast beam search                    | 3.54       | 8.91       | --epoch 30 --avg 10  | streaming            |
| modified beam search                | 3.43       | 8.86       | --epoch 30 --avg 10  | simulated streaming  |
| modified beam search                | 3.48       | 8.88       | --epoch 30 --avg 10  | streaming            |

The training command is:

```bash
./conv_emformer_transducer_stateless2/train.py \
  --world-size 6 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --full-libri 1 \
  --max-duration 280 \
  --master-port 12321 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/W5MpxekiQLSPyM4fe5hbKg/>

The simulated streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The simulated streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The simulated streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

The streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05>

##### With higher latency setup, training on full librispeech

In this model, the lengths of chunk and right context are 64 frames (i.e., 0.64s) and 16 frames (i.e., 0.16s), respectively.

The WERs are:

|                                     | test-clean | test-other | comment              | decoding mode        |
|-------------------------------------|------------|------------|----------------------|----------------------|
| greedy search (max sym per frame 1) | 3.3        | 8.71       | --epoch 30 --avg 10  | simulated streaming  |
| greedy search (max sym per frame 1) | 3.35       | 8.65       | --epoch 30 --avg 10  | streaming            |
| fast beam search                    | 3.27       | 8.58       | --epoch 30 --avg 10  | simulated streaming  |
| fast beam search                    | 3.31       | 8.48       | --epoch 30 --avg 10  | streaming            |
| modified beam search                | 3.26       | 8.56       | --epoch 30 --avg 10  | simulated streaming  |
| modified beam search                | 3.29       | 8.47       | --epoch 30 --avg 10  | streaming            |

The training command is:

```bash
./conv_emformer_transducer_stateless2/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --full-libri 1 \
  --max-duration 280 \
  --master-port 12321 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/eRx6XwbOQhGlywgD8lWBjw/>

The simulated streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The simulated streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The simulated streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless2/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

The streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless2/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless2/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 64 \
  --cnn-module-kernel 31 \
  --left-context-length 64 \
  --right-context-length 16 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-larger-latency-2022-07-06>


### LibriSpeech BPE training results (Pruned Stateless Streaming Conformer RNN-T)

#### [pruned_transducer_stateless](./pruned_transducer_stateless)

See <https://github.com/k2-fsa/icefall/pull/380> for more details.

##### Training on full librispeech
The WERs are (the number in the table formatted as test-clean & test-other):

We only trained 25 epochs for saving time, if you want to get better results you can train more epochs.

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 4.74 & 11.38   | 4.57 & 10.86   | 4.18 & 10.37   | 3.87 & 9.85    |
| greedy search        | 64           | 4.74 & 11.25   | 4.48 & 10.72   | 4.1 & 10.24    | 3.85 & 9.73    |
| fast beam search     | 32           | 4.75 & 11.1    | 4.48 & 10.65   | 4.12 & 10.18   | 3.95 & 9.67    |
| fast beam search     | 64           | 4.7 & 11       | 4.37 & 10.49   | 4.07 & 10.04   | 3.89 & 9.53    |
| modified beam search | 32           | 4.64 & 10.94   | 4.38 & 10.51   | 4.11 & 10.14   | 3.87 & 9.61    |
| modified beam search | 64           | 4.59 & 10.81   | 4.29 & 10.39   | 4.02 & 10.02   | 3.84 & 9.43    |

**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is:

```bash
./pruned_transducer_stateless/train.py \
  --exp-dir pruned_transducer_stateless/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 20 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --start-epoch 0 \
  --num-epochs 25
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/ofxRakE6R7WHB1AoB8Bweg/>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless/decode.py \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 24 \
            --avg 10 \
            --exp-dir ./pruned_transducer_stateless/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless_20220625>

#### [pruned_transducer_stateless2](./pruned_transducer_stateless2)

See <https://github.com/k2-fsa/icefall/pull/380> for more details.

##### Training on full librispeech
The WERs are (the number in the table formatted as test-clean & test-other):

We only trained 25 epochs for saving time, if you want to get better results you can train more epochs.

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 4.2 & 10.64    | 3.97 & 10.03   | 3.83 & 9.58    | 3.7 & 9.11     |
| greedy search        | 64           | 4.16 & 10.5    | 3.93 & 9.99    | 3.73 & 9.45    | 3.63 & 9.04    |
| fast beam search     | 32           | 4.13 & 10.3    | 3.93 & 9.82    | 3.8 & 9.35     | 3.62 & 8.93    |
| fast beam search     | 64           | 4.13 & 10.22   | 3.89 & 9.68    | 3.73 & 9.27    | 3.52 & 8.82    |
| modified beam search | 32           | 4.02 & 10.22   | 3.9 & 9.71     | 3.74 & 9.33    | 3.59 & 8.87    |
| modified beam search | 64           | 4.05 & 10.08   | 3.81 & 9.67    | 3.68 & 9.21    | 3.56 & 8.77    |

**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless2/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is:

```bash
./pruned_transducer_stateless2/train.py \
  --exp-dir pruned_transducer_stateless2/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 20 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --start-epoch 0 \
  --num-epochs 25
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/hbltNS5TQ1Kiw0D1vcoakw/>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless2/decode.py \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 24 \
            --avg 10 \
            --exp-dir ./pruned_transducer_stateless2/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless2_20220625>

#### [pruned_transducer_stateless3](./pruned_transducer_stateless3)

See <https://github.com/k2-fsa/icefall/pull/380> for more details.

##### Training on full librispeech (**Use giga_prob = 0.5**)

The WERs are (the number in the table formatted as test-clean & test-other):

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 3.7 & 9.53     | 3.45 & 8.88    | 3.28 & 8.45    | 3.13 & 7.93    |
| greedy search        | 64           | 3.69 & 9.36    | 3.39 & 8.68    | 3.28 & 8.19    | 3.08 & 7.83    |
| fast beam search     | 32           | 3.71 & 9.18    | 3.36 & 8.65    | 3.23 & 8.23    | 3.17 & 7.78    |
| fast beam search     | 64           | 3.61 & 9.03    | 3.46 & 8.43    | 3.2 & 8.0      | 3.11 & 7.63    |
| modified beam search | 32           | 3.56 & 9.08    | 3.34 & 8.58    | 3.21 & 8.14    | 3.06 & 7.73    |
| modified beam search | 64           | 3.55 & 8.86    | 3.29 & 8.34    | 3.16 & 8.01    | 3.05 & 7.57    |

**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless3/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is (Note: this model was trained with mix-precision training):

```bash
./pruned_transducer_stateless3/train.py \
  --exp-dir pruned_transducer_stateless3/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 32 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --use-fp16 1 \
  --start-epoch 0 \
  --num-epochs 37 \
  --num-workers 2 \
  --giga-prob 0.5
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/vL7dWVZqTYaSeoOED4rtow/>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless3/decode.py \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 36 \
            --avg 8 \
            --exp-dir ./pruned_transducer_stateless3/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.5_20220625>

##### Training on full librispeech (**Use giga_prob = 0.9**)

The WERs are (the number in the table formatted as test-clean & test-other):

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 3.25 & 8.2     | 3.07 & 7.67    | 2.91 & 7.28    | 2.8 & 6.89     |
| greedy search        | 64           | 3.22 & 8.12    | 3.05 & 7.59    | 2.91 & 7.07    | 2.78 & 6.81    |
| fast beam search     | 32           | 3.26 & 8.2     | 3.06 & 7.56    | 2.98 & 7.08    | 2.77 & 6.75    |
| fast beam search     | 64           | 3.24 & 8.09    | 3.06 & 7.43    | 2.88 & 7.03    | 2.73 & 6.68    |
| modified beam search | 32           | 3.13 & 7.91    | 2.99 & 7.45    | 2.83 & 6.98    | 2.68 & 6.75    |
| modified beam search | 64           | 3.08 & 7.8     | 2.97 & 7.37    | 2.81 & 6.82    | 2.66 & 6.67    |

**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless3/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is:

```bash
./pruned_transducer_stateless3/train.py \
  --exp-dir pruned_transducer_stateless3/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 25 \
  --num-left-chunks 8 \
  --max-duration 300 \
  --world-size 8 \
  --start-epoch 0 \
  --num-epochs 26 \
  --num-workers 2 \
  --giga-prob 0.9
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/WBGBDzt7SByRnvCBEfQpGQ/>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless3/decode.py \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 25 \
            --avg 12 \
            --exp-dir ./pruned_transducer_stateless3/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625>

#### [pruned_transducer_stateless4](./pruned_transducer_stateless4)

See <https://github.com/k2-fsa/icefall/pull/380> for more details.

##### Training on full librispeech
The WERs are (the number in the table formatted as test-clean & test-other):

We only trained 25 epochs for saving time, if you want to get better results you can train more epochs.

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 3.96 & 10.45   | 3.73 & 9.97    | 3.54 & 9.56    | 3.45 & 9.08    |
| greedy search        | 64           | 3.9 & 10.34    | 3.7 & 9.9      | 3.53 & 9.41    | 3.39 & 9.03    |
| fast beam search     | 32           | 3.9 & 10.09    | 3.69 & 9.65    | 3.58 & 9.28    | 3.46 & 8.91    |
| fast beam search     | 64           | 3.82 & 10.03   | 3.67 & 9.56    | 3.51 & 9.18    | 3.43 & 8.78    |
| modified beam search | 32           | 3.78 & 10.0    | 3.63 & 9.54    | 3.43 & 9.29    | 3.39 & 8.84    |
| modified beam search | 64           | 3.76 & 9.95    | 3.54 & 9.48    | 3.4 & 9.13     | 3.33 & 8.74    |

**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless4/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is:

```bash
./pruned_transducer_stateless4/train.py \
  --exp-dir pruned_transducer_stateless4/exp \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 20 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 25
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/97VKXf80Ru61CnP2ALWZZg/>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless4/decode.py \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 25 \
            --avg 3 \
            --exp-dir ./pruned_transducer_stateless4/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625>

#### [pruned_transducer_stateless5](./pruned_transducer_stateless5)

See <https://github.com/k2-fsa/icefall/pull/454> for more details.

##### Training on full librispeech
The WERs are (the number in the table formatted as test-clean & test-other):

We only trained 25 epochs for saving time, if you want to get better results you can train more epochs.

| decoding method      | left context | chunk size = 2 | chunk size = 4 | chunk size = 8 | chunk size = 16|
|----------------------|--------------|----------------|----------------|----------------|----------------|
| greedy search        | 32           | 3.93 & 9.88    | 3.64 & 9.43    | 3.51 & 8.92    | 3.26 & 8.37    |
| greedy search        | 64           | 4.84 & 9.81    | 3.59 & 9.27    | 3.44 & 8.83    | 3.23 & 8.33    |
| fast beam search     | 32           | 3.86 & 9.77    | 3.67 & 9.3     | 3.5 & 8.83     | 3.27 & 8.33    |
| fast beam search     | 64           | 3.79 & 9.68    | 3.57 & 9.21    | 3.41 & 8.72    | 3.25 & 8.27    |
| modified beam search | 32           | 3.84 & 9.71    | 3.66 & 9.38    | 3.47 & 8.86    | 3.26 & 8.42    |
| modified beam search | 64           | 3.81 & 9.59    | 3.58 & 9.2     | 3.44 & 8.74    | 3.23 & 8.35    |


**NOTE:** The WERs in table above were decoded with simulate streaming method (i.e. using masking strategy), see commands below. We also have [real streaming decoding](./pruned_transducer_stateless5/streaming_decode.py) script which should produce almost the same results. We tried adding right context in the real streaming decoding, but it seemed not to benefit the performance for all the models, the reasons might be the training and decoding mismatching.

The training command is:

```bash
./pruned_transducer_stateless5/train.py \
  --exp-dir pruned_transducer_stateless5/exp \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --full-libri 1 \
  --dynamic-chunk-training 1 \
  --causal-convolution 1 \
  --short-chunk-size 20 \
  --num-left-chunks 4 \
  --max-duration 300 \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 25
```

You can find the tensorboard log here <https://tensorboard.dev/experiment/rO04h6vjTLyw0qSxjp4m4Q>

The decoding command is:
```bash
decoding_method="greedy_search"  # "fast_beam_search", "modified_beam_search"

for chunk in 2 4 8 16; do
  for left in 32 64; do
    ./pruned_transducer_stateless5/decode.py \
            --num-encoder-layers 18 \
            --dim-feedforward 2048 \
            --nhead 8 \
            --encoder-dim 512 \
            --decoder-dim 512 \
            --joiner-dim 512 \
            --simulate-streaming 1 \
            --decode-chunk-size ${chunk} \
            --left-context ${left} \
            --causal-convolution 1 \
            --epoch 25 \
            --avg 3 \
            --exp-dir ./pruned_transducer_stateless5/exp \
            --max-sym-per-frame 1 \
            --max-duration 1000 \
            --decoding-method ${decoding_method}
  done
done
```

Pre-trained models, training and decoding logs, and decoding results are available at <https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless5_20220729>


### LibriSpeech BPE training results (Pruned Stateless Conv-Emformer RNN-T)

#### [conv_emformer_transducer_stateless](./conv_emformer_transducer_stateless)

It implements [Emformer](https://arxiv.org/abs/2010.10759) augmented with convolution module for streaming ASR.
It is modified from [torchaudio](https://github.com/pytorch/audio).

See <https://github.com/k2-fsa/icefall/pull/389> for more details.

##### Training on full librispeech

In this model, the lengths of chunk and right context are 32 frames (i.e., 0.32s) and 8 frames (i.e., 0.08s), respectively.

The WERs are:

|                                     | test-clean | test-other | comment              | decoding mode        |
|-------------------------------------|------------|------------|----------------------|----------------------|
| greedy search (max sym per frame 1) | 3.63       | 9.61       | --epoch 30 --avg 10  | simulated streaming  |
| greedy search (max sym per frame 1) | 3.64       | 9.65       | --epoch 30 --avg 10  | streaming            |
| fast beam search                    | 3.61       | 9.4        | --epoch 30 --avg 10  | simulated streaming  |
| fast beam search                    | 3.58       | 9.5        | --epoch 30 --avg 10  | streaming            |
| modified beam search                | 3.56       | 9.41       | --epoch 30 --avg 10  | simulated streaming  |
| modified beam search                | 3.54       | 9.46       | --epoch 30 --avg 10  | streaming            |

The training command is:

```bash
./conv_emformer_transducer_stateless/train.py \
  --world-size 6 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 300 \
  --master-port 12321 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/4em2FLsxRwGhmoCRQUEoDw/>

The simulated streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The simulated streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The simulated streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless/decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --max-duration 300 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

The streaming decoding command using greedy search is:
```bash
./conv_emformer_transducer_stateless/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The streaming decoding command using fast beam search is:
```bash
./conv_emformer_transducer_stateless/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The streaming decoding command using modified beam search is:
```bash
./conv_emformer_transducer_stateless/streaming_decode.py \
  --epoch 30 \
  --avg 10 \
  --exp-dir conv_emformer_transducer_stateless/exp \
  --num-decode-streams 2000 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless-2022-06-11>

### LibriSpeech BPE training results (Pruned Stateless Emformer RNN-T)

#### [pruned_stateless_emformer_rnnt2](./pruned_stateless_emformer_rnnt2)

Use <https://github.com/k2-fsa/icefall/pull/390>.

Use [Emformer](https://arxiv.org/abs/2010.10759) from [torchaudio](https://github.com/pytorch/audio)
for streaming ASR. The Emformer model is imported from torchaudio without modifications.

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

|                                     | test-clean | test-other | comment                                |
|-------------------------------------|------------|------------|----------------------------------------|
| greedy search (max sym per frame 1) | 4.28       | 11.42       | --epoch 39 --avg 6  --max-duration 600 |
| modified beam search                | 4.22       | 11.16       | --epoch 39 --avg 6  --max-duration 600 |
| fast beam search                    | 4.29       | 11.26       | --epoch 39 --avg 6 --max-duration 600  |


The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_stateless_emformer_rnnt2/train.py \
  --world-size 8 \
  --num-epochs 40 \
  --start-epoch 1 \
  --exp-dir pruned_stateless_emformer_rnnt2/exp-full \
  --full-libri 1 \
  --use-fp16 0 \
  --max-duration 200 \
  --prune-range 5 \
  --lm-scale 0.25 \
  --master-port 12358 \
  --num-encoder-layers 18 \
  --left-context-length 128 \
  --segment-length 8 \
  --right-context-length 4
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/ZyiqhAhmRjmr49xml4ofLw/>

The decoding commands are:
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  for epoch in 39; do
    for avg in 6; do
      ./pruned_stateless_emformer_rnnt2/decode.py \
        --epoch $epoch \
        --avg $avg \
        --use-averaged-model 1 \
        --exp-dir pruned_stateless_emformer_rnnt2/exp-full \
        --max-duration 50 \
        --decoding-method $m \
        --num-encoder-layers 18 \
        --left-context-length 128 \
        --segment-length 8 \
        --right-context-length 4
    done
  done
done
```

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01>


### LibriSpeech BPE training results (Pruned Stateless Transducer 5)

#### [pruned_transducer_stateless5](./pruned_transducer_stateless5)

Same as `Pruned Stateless Transducer 2` but with more layers.

See <https://github.com/k2-fsa/icefall/pull/330>

Note that models in `pruned_transducer_stateless` and `pruned_transducer_stateless2`
have about 80 M parameters.

The notations `large` and `medium` below are from the [Conformer](https://arxiv.org/pdf/2005.08100.pdf)
paper, where the large model has about 118 M parameters and the medium model
has 30.8 M parameters.

##### Large

Number of model parameters 118129516 (i.e, 118.13 M).

|                                     | test-clean | test-other | comment                                |
|-------------------------------------|------------|------------|----------------------------------------|
| greedy search (max sym per frame 1) | 2.43       | 5.72       | --epoch 30 --avg 10 --max-duration 600 |
| modified beam search                | 2.43       | 5.69       | --epoch 30 --avg 10 --max-duration 600 |
| fast beam search                    | 2.43       | 5.67       | --epoch 30 --avg 10 --max-duration 600 |

The training commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless5/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/exp-L \
  --max-duration 300 \
  --use-fp16 0 \
  --num-encoder-layers 18 \
  --dim-feedforward 2048 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 512
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/aWzDj5swSE2VmcOYgoe3vQ>

The decoding commands are:

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-L \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 18 \
    --dim-feedforward 2048 \
    --nhead 8 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-averaged-model True
done
```

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless5-2022-07-07>


##### Medium

Number of model parameters 30896748 (i.e, 30.9 M).

|                                     | test-clean | test-other | comment                                 |
|-------------------------------------|------------|------------|-----------------------------------------|
| greedy search (max sym per frame 1) | 2.87       | 6.92       | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search                | 2.83       | 6.75       | --epoch 30 --avg 10  --max-duration 600 |
| fast beam search                    | 2.81       | 6.76       | --epoch 30 --avg 10  --max-duration 600  |

The training commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless5/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/exp-M \
  --max-duration 300 \
  --use-fp16 0 \
  --num-encoder-layers 18 \
  --dim-feedforward 1024 \
  --nhead 4 \
  --encoder-dim 256 \
  --decoder-dim 512 \
  --joiner-dim 512
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/04xtWUKPRmebSnpzN1GMHQ>

The decoding commands are:

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-M \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 18 \
    --dim-feedforward 1024 \
    --nhead 4 \
    --encoder-dim 256 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-averaged-model True
done
```

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless5-M-2022-07-07>


##### Baseline-2

It has 87.8 M parameters. Compared to the model in pruned_transducer_stateless2, its has more
layers (24 v.s 12) but a narrower model (1536 feedforward dim and 384 encoder dim vs 2048 feed forward dim and 512 encoder dim).

|                                     | test-clean | test-other | comment                                 |
|-------------------------------------|------------|------------|-----------------------------------------|
| greedy search (max sym per frame 1) | 2.54       | 5.72       | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search                | 2.47       | 5.71       | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search + RNNLM shallow fusion     | 2.27       | 5.24      | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search + RNNLM + LODR     | 2.23       | 5.17      | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search + TransformerLM shallow fusion     | 2.27       | 5.26      | --epoch 30 --avg 10  --max-duration 600 |
| modified beam search + TransformerLM + LODR     | 2.22       | 5.11      | --epoch 30 --avg 10  --max-duration 600 |
| fast beam search                    | 2.5        | 5.72       | --epoch 30 --avg 10  --max-duration 600 |

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless5/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless5/exp-B \
  --max-duration 300 \
  --use-fp16 0 \
  --num-encoder-layers 24 \
  --dim-feedforward 1536 \
  --nhead 8 \
  --encoder-dim 384 \
  --decoder-dim 512 \
  --joiner-dim 512
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/foVHNyqiRi2LhybmRUOAyg>

The decoding commands are:

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-B \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 24 \
    --dim-feedforward 1536 \
    --nhead 8 \
    --encoder-dim 384 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-averaged-model True
done
```

To decode with RNNLM shallow fusion, use the following decoding command. A well-trained RNNLM
can be found here: <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-B \
    --max-duration 600 \
    --decoding-method modified_beam_search_rnnlm_shallow_fusion \
    --max-sym-per-frame 1 \
    --num-encoder-layers 24 \
    --dim-feedforward 1536 \
    --nhead 8 \
    --encoder-dim 384 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --use-averaged-model True
    --beam 4 \
    --max-contexts 4 \
    --rnn-lm-scale 0.4 \
    --rnn-lm-exp-dir /path/to/RNNLM/exp \
    --rnn-lm-epoch 99 \
    --rnn-lm-avg 1 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights 1
done
```

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless5-B-2022-07-07>


### LibriSpeech BPE training results (Pruned Stateless Transducer 4)

#### [pruned_transducer_stateless4](./pruned_transducer_stateless4)

This version saves averaged model during training, and decodes with averaged model.

See <https://github.com/k2-fsa/icefall/issues/337> for details about the idea of model averaging.

##### Training on full librispeech

See <https://github.com/k2-fsa/icefall/pull/344>

Using commit `ec0b0e92297cc03fdb09f48cd235e84d2c04156b`.

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|-------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 2.75       | 6.74       | --epoch 30 --avg 6  --use-averaged-model False                                |
| greedy search (max sym per frame 1) | 2.69       | 6.64       | --epoch 30 --avg 6  --use-averaged-model True                                 |
| fast beam search                    | 2.72       | 6.67       | --epoch 30 --avg 6  --use-averaged-model False                                |
| fast beam search                    | 2.66       | 6.6        | --epoch 30 --avg 6  --use-averaged-model True                                 |
| modified beam search                | 2.67       | 6.68       | --epoch 30 --avg 6  --use-averaged-model False                                |
| modified beam search                | 2.62       | 6.57       | --epoch 30 --avg 6  --use-averaged-model True                                 |

The training command is:

```bash
./pruned_transducer_stateless4/train.py \
  --world-size 6 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless4/exp \
  --full-libri 1 \
  --max-duration 300 \
  --save-every-n 8000 \
  --keep-last-k 20 \
  --average-period 100
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/QOGSPBgsR8KzcRMmie9JGw/>

The decoding command using greedy search is:
```bash
./pruned_transducer_stateless4/decode.py \
  --epoch 30 \
  --avg 6 \
  --exp-dir pruned_transducer_stateless4/exp \
  --max-duration 300 \
  --decoding-method greedy_search \
  --use-averaged-model True
```

The decoding command using fast beam search is:
```bash
./pruned_transducer_stateless4/decode.py \
  --epoch 30 \
  --avg 6 \
  --exp-dir pruned_transducer_stateless4/exp \
  --max-duration 300 \
  --decoding-method fast_beam_search \
  --use-averaged-model True \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8
```

The decoding command using modified beam search is:
```bash
./pruned_transducer_stateless4/decode.py \
  --epoch 30 \
  --avg 6 \
  --exp-dir pruned_transducer_stateless4/exp \
  --max-duration 300 \
  --decoding-method modified_beam_search \
  --use-averaged-model True \
  --beam-size 4
```

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless4-2022-06-03>

##### Training on train-clean-100

See <https://github.com/k2-fsa/icefall/pull/344>

Using commit `ec0b0e92297cc03fdb09f48cd235e84d2c04156b`.

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|-------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 7.0        | 18.95      | --epoch 30 --avg 10 --use_averaged_model False                                |
| greedy search (max sym per frame 1) | 6.92       | 18.65      | --epoch 30 --avg 10 --use_averaged_model True                                 |
| fast beam search                    | 6.82       | 18.47      | --epoch 30 --avg 10 --use_averaged_model False                                |
| fast beam search                    | 6.74       | 18.2       | --epoch 30 --avg 10 --use_averaged_model True                                 |
| modified beam search                | 6.74       | 18.39      | --epoch 30 --avg 10 --use_averaged_model False                                |
| modified beam search                | 6.74       | 18.12      | --epoch 30 --avg 10 --use_averaged_model True                                 |

The training command is:

```bash
./pruned_transducer_stateless4/train.py \
  --world-size 3 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless4/exp \
  --full-libri 0 \
  --max-duration 300 \
  --save-every-n 8000 \
  --keep-last-k 20 \
  --average-period 100
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/YVYHq1irQS69s9bW1vQ06Q/>

### LibriSpeech BPE training results (Pruned Stateless Transducer 3, 2022-04-29)

#### [pruned_transducer_stateless3](./pruned_transducer_stateless3)
Same as `Pruned Stateless Transducer 2` but using the XL subset from
[GigaSpeech](https://github.com/SpeechColab/GigaSpeech) as extra training data.

During training, it selects either a batch from GigaSpeech with prob `giga_prob`
or a batch from LibriSpeech with prob `1 - giga_prob`. All utterances within
a batch come from the same dataset.

Using commit `ac84220de91dee10c00e8f4223287f937b1930b6`.

See <https://github.com/k2-fsa/icefall/pull/312>.

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|----------------------------------------|
| greedy search (max sym per frame 1) | 2.21       | 5.09       | --epoch 27 --avg 2  --max-duration 600 |
| greedy search (max sym per frame 1) | 2.25       | 5.02       | --epoch 27 --avg 12 --max-duration 600 |
| modified beam search                | 2.19       | 5.03       | --epoch 25 --avg 6  --max-duration 600 |
| modified beam search                | 2.23       | 4.94       | --epoch 27 --avg 10 --max-duration 600 |
| beam search                         | 2.16       | 4.95       | --epoch 25 --avg 7  --max-duration 600 |
| fast beam search                    | 2.21       | 4.96       | --epoch 27 --avg 10 --max-duration 600 |
| fast beam search                    | 2.19       | 4.97       | --epoch 27 --avg 12 --max-duration 600 |

The training commands are:

```bash
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless3/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 0 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless3/exp \
  --max-duration 300 \
  --use-fp16 1 \
  --lr-epochs 4 \
  --num-workers 2 \
  --giga-prob 0.8
```

The tensorboard log can be found at
<https://tensorboard.dev/experiment/gaD34WeYSMCOkzoo3dZXGg/>
(Note: The training process is killed manually after saving `epoch-28.pt`.)

Pretrained models, training logs, decoding logs, and decoding results
are available at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-04-29>

The decoding commands are:

```bash

# greedy search
./pruned_transducer_stateless3/decode.py \
    --epoch 27 \
    --avg 2 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method greedy_search \
    --max-sym-per-frame 1

# modified beam search
./pruned_transducer_stateless3/decode.py \
    --epoch 25 \
    --avg 6 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method modified_beam_search \
    --max-sym-per-frame 1

# beam search
./pruned_transducer_stateless3/decode.py \
    --epoch 25 \
    --avg 7 \
    --exp-dir ./pruned_transducer_stateless3/exp \
    --max-duration 600 \
    --decoding-method beam_search \
    --max-sym-per-frame 1

# fast beam search
for epoch in 27; do
  for avg in 10 12; do
    ./pruned_transducer_stateless3/decode.py \
        --epoch $epoch \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp \
        --max-duration 600 \
        --decoding-method fast_beam_search \
        --max-states 32 \
        --beam 8
  done
done
```

The following table shows the
[Nbest oracle WER](http://kaldi-asr.org/doc/lattices.html#lattices_operations_oracle)
for fast beam search.

| epoch | avg | num_paths | nbest_scale | test-clean | test-other |
|-------|-----|-----------|-------------|------------|------------|
|  27   | 10  |   50      | 0.5         |  0.91      |  2.74      |
|  27   | 10  |   50      | 0.8         |  0.94      |  2.82      |
|  27   | 10  |   50      | 1.0         |  1.06      |  2.88      |
|  27   | 10  |   100     | 0.5         |  0.82      |  2.58      |
|  27   | 10  |   100     | 0.8         |  0.92      |  2.65      |
|  27   | 10  |   100     | 1.0         |  0.95      |  2.77      |
|  27   | 10  |   200     | 0.5         |  0.81      |  2.50      |
|  27   | 10  |   200     | 0.8         |  0.85      |  2.56      |
|  27   | 10  |   200     | 1.0         |  0.91      |  2.64      |
|  27   | 10  |   400     | 0.5         |  N/A       |  N/A       |
|  27   | 10  |   400     | 0.8         |  0.81      |  2.49      |
|  27   | 10  |   400     | 1.0         |  0.85      |  2.54      |

The Nbest oracle WER is computed using the following steps:

  - 1. Use `fast_beam_search` to produce a lattice.
  - 2. Extract `N` paths from the lattice using [k2.random_path](https://k2-fsa.github.io/k2/python_api/api.html#random-paths)
  - 3. [Unique](https://k2-fsa.github.io/k2/python_api/api.html#unique) paths so that each path
       has a distinct sequence of tokens
  - 4. Compute the edit distance of each path with the ground truth
  - 5. The path with the lowest edit distance is the final output and is used to
       compute the WER

The command to compute the Nbest oracle WER is:

```bash
for epoch in 27; do
  for avg in 10 ; do
    for num_paths in 50 100 200 400; do
      for nbest_scale in 0.5 0.8 1.0; do
        ./pruned_transducer_stateless3/decode.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir ./pruned_transducer_stateless3/exp \
            --max-duration 600 \
            --decoding-method fast_beam_search_nbest_oracle \
            --num-paths $num_paths \
            --max-states 32 \
            --beam 8 \
            --nbest-scale $nbest_scale
      done
    done
  done
done
```

### LibriSpeech BPE training results (Pruned Transducer 3, 2022-05-13)

Same setup as [pruned_transducer_stateless3](./pruned_transducer_stateless3) (2022-04-29)
but change `--giga-prob` from 0.8 to 0.9. Also use `repeat` on gigaspeech XL
subset so that the gigaspeech dataloader never exhausts.

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|---------------------------------------------|
| greedy search (max sym per frame 1) | 2.03       | 4.70       | --iter 1224000 --avg 14  --max-duration 600 |
| modified beam search                | 2.00       | 4.63       | --iter 1224000 --avg 14  --max-duration 600 |
| modified beam search + rnnlm shallow fusion  | 1.94     |  4.2    | --iter 1224000 --avg 14  --max-duration 600 |
| modified beam search + rnnlm + LODR         | 1.77       | 3.99       | --iter 1224000 --avg 14  --max-duration 600 |
| modified beam search + TransformerLM + LODR    | 1.75       | 3.94       | --iter 1224000 --avg 14  --max-duration 600 |
| fast beam search                    | 2.10       | 4.68       | --iter 1224000 --avg 14 --max-duration 600 |

The training commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./prepare.sh
./prepare_giga_speech.sh

./pruned_transducer_stateless3/train.py \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 0 \
  --full-libri 1 \
  --exp-dir pruned_transducer_stateless3/exp-0.9 \
  --max-duration 300 \
  --use-fp16 1 \
  --lr-epochs 4 \
  --num-workers 2 \
  --giga-prob 0.9
```

The tensorboard log is available at
<https://tensorboard.dev/experiment/HpocR7dKS9KCQkJeYxfXug/>

Decoding commands:

```bash
for iter in 1224000; do
  for avg in 14; do
    for method in greedy_search modified_beam_search fast_beam_search ; do
      ./pruned_transducer_stateless3/decode.py \
        --iter $iter \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp-0.9/ \
        --max-duration 600 \
        --decoding-method $method \
        --max-sym-per-frame 1 \
        --beam 4 \
        --max-contexts 32
    done
  done
done
```
You may also decode using shallow fusion with external neural network LM. To do so you need to
download a well-trained NN LM:
RNN LM: <https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>
Transformer LM: <https://huggingface.co/marcoyang/icefall-librispeech-transformer-lm/tree/main>

```bash
rnn_lm_scale=0.3

for iter in 1224000; do
  for avg in 14; do
    for method in modified_beam_search_rnnlm_shallow_fusion ; do
      ./pruned_transducer_stateless3/decode.py \
        --iter $iter \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp-0.9/ \
        --max-duration 600 \
        --decoding-method $method \
        --max-sym-per-frame 1 \
        --beam 4 \
        --max-contexts 32 \
        --rnn-lm-scale $rnn_lm_scale \
        --rnn-lm-exp-dir /path/to/RNNLM \
        --rnn-lm-epoch 99 \
        --rnn-lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --rnn-lm-tie-weights 1
    done
  done
done
```

If you want to try out with LODR decoding, use the following command. This assums you have a bi-gram LM trained on LibriSpeech text. You can also download the bi-gram LM from here <https://huggingface.co/marcoyang/librispeech_bigram/tree/main> and put it under the directory `data/lang_bpe_500`.

```bash
rnn_lm_scale=0.4

for iter in 1224000; do
  for avg in 14; do
    for method in modified_beam_search_rnnlm_LODR ; do
      ./pruned_transducer_stateless3/decode.py \
        --iter $iter \
        --avg $avg \
        --exp-dir ./pruned_transducer_stateless3/exp-0.9/ \
        --max-duration 600 \
        --decoding-method $method \
        --max-sym-per-frame 1 \
        --beam 4 \
        --max-contexts 32 \
        --rnn-lm-scale $rnn_lm_scale \
        --rnn-lm-exp-dir /path/to/RNNLM \
        --rnn-lm-epoch 99 \
        --rnn-lm-avg 1 \
        --rnn-lm-num-layers 3 \
        --rnn-lm-tie-weights 1 \
        --tokens-ngram 2 \
        --ngram-lm-scale -0.14
    done
  done
done
```

The pretrained models, training logs, decoding logs, and decoding results
can be found at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>


### LibriSpeech BPE training results (Pruned Transducer 2)

#### [pruned_transducer_stateless2](./pruned_transducer_stateless2)
This is with a reworked version of the conformer encoder, with many changes.

##### Training on full librispeech

Using commit `34aad74a2c849542dd5f6359c9e6b527e8782fd6`.
See <https://github.com/k2-fsa/icefall/pull/288>

The WERs are:

|                                     | test-clean | test-other | comment                                                                       |
|-------------------------------------|------------|------------|-------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 2.62       | 6.37       | --epoch 25 --avg 8  --max-duration 600                                        |
| fast beam search                    | 2.61       | 6.17       | --epoch 25 --avg 8  --max-duration 600 --decoding-method fast_beam_search     |
| modified beam search                | 2.59       | 6.19       | --epoch 25 --avg 8  --max-duration 600 --decoding-method modified_beam_search |
| greedy search (max sym per frame 1) | 2.70       | 6.04       | --epoch 34 --avg 10 --max-duration 600                                        |
| fast beam search                    | 2.66       | 6.00       | --epoch 34 --avg 10  --max-duration 600 --decoding-method fast_beam_search    |
| greedy search (max sym per frame 1) | 2.62       | 6.03       | --epoch 38 --avg 10 --max-duration 600                                        |
| fast beam search                    | 2.57       | 5.95       | --epoch 38 --avg 10  --max-duration 600 --decoding-method fast_beam_search    |




The train and decode commands are:
```bash
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp \
  --world-size 8 \
  --num-epochs 26 \
  --full-libri 1 \
  --max-duration 300
```

and:

```bash
python3 ./pruned_transducer_stateless2/decode.py \
  --exp-dir pruned_transducer_stateless2/exp \
  --epoch 25 \
  --avg 8 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 600
```

The Tensorboard log is at <https://tensorboard.dev/experiment/Xoz0oABMTWewo1slNFXkyA> (apologies, log starts
only from epoch 3).

The pretrained models, training logs, decoding logs, and decoding results
can be found at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless2-2022-04-29>


##### Training on train-clean-100:

Trained with 1 job:
```
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp_100h_ws1 \
  --world-size 1 \
  --num-epochs 40  \
  --full-libri 0 \
  --max-duration 300
```

and decoded with:

```
python3 ./pruned_transducer_stateless2/decode.py \
  --exp-dir pruned_transducer_stateless2/exp_100h_ws1 \
  --epoch 19 \
  --avg 8 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 600
```


The Tensorboard log is at <https://tensorboard.dev/experiment/AhnhooUBRPqTnaggoqo7lg> (learning rate
schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment                                               |
|-------------------------------------|------------|------------|-------------------------------------------------------|
| greedy search (max sym per frame 1) | 7.12       | 18.42      | --epoch 19 --avg 8                                    |
| greedy search (max sym per frame 1) | 6.71       | 17.77      | --epoch 29 --avg 8                                    |
| greedy search (max sym per frame 1) | 6.64       | 17.19      | --epoch 39 --avg 10                                    |
| fast beam search                    | 6.58       | 17.27      | --epoch 29 --avg 8 --decoding-method fast_beam_search |
| fast beam search                    | 6.53       | 16.82      | --epoch 39 --avg 10 --decoding-method fast_beam_search |

Trained with 2 jobs:

```bash
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp_100h_ws2 \
  --world-size 2 \
  --num-epochs 40  \
  --full-libri 0 \
  --max-duration 300
```

and decoded with:

```
python3 ./pruned_transducer_stateless2/decode.py \
  --exp-dir pruned_transducer_stateless2/exp_100h_ws2 \
  --epoch 19 \
  --avg 8 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 600
```

The Tensorboard log is at <https://tensorboard.dev/experiment/dvOC9wsrSdWrAIdsebJILg/>
(learning rate schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.05       | 18.77      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.82       | 18.14      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.81       | 17.66      | --epoch 30  --avg 10  |


Trained with 4 jobs:

```
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp_100h_ws4 \
  --world-size 4 \
  --num-epochs 40  \
  --full-libri 0 \
  --max-duration 300
```

and decoded with:

```
python3 ./pruned_transducer_stateless2/decode.py \
  --exp-dir pruned_transducer_stateless2/exp_100h_ws4 \
  --epoch 19 \
  --avg 8 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 600
```


The Tensorboard log is at <https://tensorboard.dev/experiment/a3T0TyC0R5aLj5bmFbRErA/>
(learning rate schedule is not visible due to a since-fixed bug).

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.31       | 19.55      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 7.08       | 18.59      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.86       | 18.29      | --epoch 30  --avg 10  |



Trained with 1 job, with  --use-fp16=True --max-duration=300 i.e. with half-precision
floats (but without increasing max-duration), after merging <https://github.com/k2-fsa/icefall/pull/305>.
Train command was

```
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp_100h_fp16 \
  --world-size 1 \
  --num-epochs 40  \
  --full-libri 0 \
  --max-duration 300 \
  --use-fp16 True
```

The Tensorboard log is at <https://tensorboard.dev/experiment/DAtGG9lpQJCROUDwPNxwpA>

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.10       | 18.57      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.81       | 17.84      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.63       | 17.39      | --epoch 30  --avg 10  |


Trained with 1 job, with  --use-fp16=True --max-duration=500, i.e. with half-precision
floats and max-duration increased from 300 to 500, after merging <https://github.com/k2-fsa/icefall/pull/305>.
Train command was

```
python3 ./pruned_transducer_stateless2/train.py \
  --exp-dir=pruned_transducer_stateless2/exp_100h_fp16 \
  --world-size 1 \
  --num-epochs 40  \
  --full-libri 0 \
  --max-duration 500 \
  --use-fp16 True
```

The Tensorboard log is at <https://tensorboard.dev/experiment/Km7QBHYnSLWs4qQnAJWsaA>

|                                     | test-clean | test-other | comment               |
|-------------------------------------|------------|------------|-----------------------|
| greedy search (max sym per frame 1) | 7.10       | 18.79      | --epoch 19  --avg 8   |
| greedy search (max sym per frame 1) | 6.92       | 18.16      | --epoch 29  --avg 8   |
| greedy search (max sym per frame 1) | 6.89       | 17.75      | --epoch 30  --avg 10  |



### LibriSpeech BPE training results (Pruned Transducer)

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer, a Conv1d (with kernel size 2) and a linear
layer (to transform tensor dim).

#### 2022-03-12

[pruned_transducer_stateless](./pruned_transducer_stateless)

Using commit `1603744469d167d848e074f2ea98c587153205fa`.
See <https://github.com/k2-fsa/icefall/pull/248>

The WERs are:

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| greedy search (max sym per frame 2) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| greedy search (max sym per frame 3) | 2.62       | 6.37       | --epoch 42  --avg 11  --max-duration 100 |
| modified beam search (beam size 4)  | 2.56       | 6.27       | --epoch 42  --avg 11  --max-duration 100 |
| beam search (beam size 4)           | 2.57       | 6.27       | --epoch 42  --avg 11  --max-duration 100 |





The decoding time for `test-clean` and `test-other` is given below:
(A V100 GPU with 32 GB RAM is used for decoding. Note: Not all GPU RAM is used during decoding.)

| decoding method | test-clean (seconds) | test-other (seconds)|
|---|---:|---:|
| greedy search (--max-sym-per-frame=1) | 160 | 159 |
| greedy search (--max-sym-per-frame=2) | 184 | 177 |
| greedy search (--max-sym-per-frame=3) | 210 | 213 |
| modified beam search (--beam-size 4)| 273 | 269 |
|beam search (--beam-size 4) | 2741 | 2221 |

We recommend you to use `modified_beam_search`.

Training command:

```bash
cd egs/librispeech/ASR/
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

. path.sh

./pruned_transducer_stateless/train.py \
  --world-size 8 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 300 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/WKRFY5fYSzaVBHahenpNlA/>

The command for decoding is:

```bash
epoch=42
avg=11
sym=1

# greedy search

./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method greedy_search \
  --beam-size 4 \
  --max-sym-per-frame $sym

# modified beam search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method modified_beam_search \
  --beam-size 4

# beam search
# (not recommended)
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless/exp \
  --max-duration 100 \
  --decoding-method beam_search \
  --beam-size 4
```

You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless-2022-03-12>

#### 2022-02-18

[pruned_transducer_stateless](./pruned_transducer_stateless)


The WERs are

|                           | test-clean | test-other | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             | 2.85       | 6.98       | --epoch 28  --avg 15  --max-duration 100 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless/exp \
  --full-libri 1 \
  --max-duration 300 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25 \
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/ejG7VpakRYePNNj6AbDEUw/#scalars>

The decoding command is:
```
epoch=28
avg=15

## greedy search
./pruned_transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless/exp \
  --max-duration 100
```


### LibriSpeech BPE training results (Transducer)

#### Conformer encoder + embedding decoder

Conformer encoder + non-recurrent decoder. The decoder
contains only an embedding layer and a Conv1d (with kernel size 2).

See

- [./transducer_stateless](./transducer_stateless)
- [./transducer_stateless_multi_datasets](./transducer_stateless_multi_datasets)

##### 2022-03-01

Using commit `2332ba312d7ce72f08c7bac1e3312f7e3dd722dc`.

It uses [GigaSpeech](https://github.com/SpeechColab/GigaSpeech)
as extra training data. 20% of the time it selects a batch from L subset of
GigaSpeech and 80% of the time it selects a batch from LibriSpeech.

The WERs are

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.64       | 6.55       | --epoch 39  --avg 15  --max-duration 100 |
| modified beam search (beam size 4)  | 2.61       | 6.46       | --epoch 39  --avg 15  --max-duration 100 |

The training command for reproducing is given below:

```bash
cd egs/librispeech/ASR/
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer_stateless_multi_datasets/train.py \
  --world-size 4 \
  --num-epochs 40 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25 \
  --giga-prob 0.2
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/xmo5oCgrRVelH9dCeOkYBg/>

The decoding command is:

```bash
epoch=39
avg=15
sym=1

# greedy search
./transducer_stateless_multi_datasets/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --max-sym-per-frame $sym

# modified beam search
./transducer_stateless_multi_datasets/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless_multi_datasets/exp-full-2 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

You can find a pretrained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-multi-datasets-bpe-500-2022-03-01>


##### 2022-04-19

[transducer_stateless2](./transducer_stateless2)

This version uses torchaudio's RNN-T loss.

Using commit `fce7f3cd9a486405ee008bcbe4999264f27774a3`.
See <https://github.com/k2-fsa/icefall/pull/316>

|                                     | test-clean | test-other | comment                                                                        |
|-------------------------------------|------------|------------|--------------------------------------------------------------------------------|
| greedy search (max sym per frame 1) | 2.65       | 6.30       | --epoch 59 --avg 10  --max-duration 600                                        |
| greedy search (max sym per frame 2) | 2.62       | 6.23       | --epoch 59 --avg 10  --max-duration 100                                        |
| greedy search (max sym per frame 3) | 2.62       | 6.23       | --epoch 59 --avg 10  --max-duration 100                                        |
| modified beam search                | 2.63       | 6.15       | --epoch 59 --avg 10  --max-duration 100 --decoding-method modified_beam_search |
| beam search                         | 2.59       | 6.15       | --epoch 59 --avg 10  --max-duration 100 --decoding-method beam_search          |

**Note**: This model is trained with standard RNN-T loss. Neither modified transducer nor pruned RNN-T is used.
You can see that there is a performance degradation in WER when we limit the max symbol per frame to 1.

The number of active paths in `modified_beam_search` and `beam_search` is 4.

The training and decoding commands are:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./transducer_stateless2/train.py \
  --world-size 8 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir transducer_stateless2/exp-2 \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5

epoch=59
avg=10
# greedy search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 600 \
  --decoding-method greedy_search \
  --max-sym-per-frame 1

# modified beam search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 100 \
  --decoding-method modified_beam_search \

# beam search
./transducer_stateless2/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./transducer_stateless2/exp-2 \
  --max-duration 100 \
  --decoding-method beam_search \
```

The tensorboard log is at <https://tensorboard.dev/experiment/oAlle3dxQD2EY8ePwjIGuw/>.


You can find a pre-trained model, decoding logs, and decoding results at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless2-torchaudio-2022-04-19>



##### 2022-02-07

Using commit `a8150021e01d34ecbd6198fe03a57eacf47a16f2`.


The WERs are

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| greedy search (max sym per frame 2) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| greedy search (max sym per frame 3) | 2.67       | 6.67       | --epoch 63  --avg 19  --max-duration 100 |
| modified beam search (beam size 4)  | 2.67       | 6.57       | --epoch 63  --avg 19  --max-duration 100 |


The training command for reproducing is given below:

```
cd egs/librispeech/ASR/
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./transducer_stateless/train.py \
  --world-size 4 \
  --num-epochs 76 \
  --start-epoch 0 \
  --exp-dir transducer_stateless/exp-full \
  --full-libri 1 \
  --max-duration 300 \
  --lr-factor 5 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/qgvWkbF2R46FYA6ZMNmOjA/#scalars>

The decoding command is:
```
epoch=63
avg=19

## greedy search
for sym in 1 2 3; do
  ./transducer_stateless/decode.py \
    --epoch $epoch \
    --avg $avg \
    --exp-dir transducer_stateless/exp-full \
    --bpe-model ./data/lang_bpe_500/bpe.model \
    --max-duration 100 \
    --max-sym-per-frame $sym
done

## modified beam search

./transducer_stateless/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless/exp-full \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

You can find a pretrained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-transducer-stateless-bpe-500-2022-02-07>


#### Conformer encoder + LSTM decoder
Using commit `8187d6236c2926500da5ee854f758e621df803cc`.

Conformer encoder + LSTM decoder.

The best WER is

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 3.07       | 7.51       |

using `--epoch 34 --avg 11` with **greedy search**.

The training command to reproduce the above WER is:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./transducer/train.py \
  --world-size 4 \
  --num-epochs 35 \
  --start-epoch 0 \
  --exp-dir transducer/exp-lr-2.5-full \
  --full-libri 1 \
  --max-duration 180 \
  --lr-factor 2.5
```

The decoding command is:

```
epoch=34
avg=11

./transducer/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer/exp-lr-2.5-full \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100
```

You can find the tensorboard log at: <https://tensorboard.dev/experiment/D7NQc3xqTpyVmWi5FnWjrA>


### LibriSpeech BPE training results (Conformer-CTC 2)

#### [conformer_ctc2](./conformer_ctc2)

#### 2022-07-21

It implements a 'reworked' version of CTC attention model.
As demenstrated by pruned_transducer_stateless2, reworked Conformer model has superior performance compared to the original Conformer.
So in this modified version of CTC attention model, it has the reworked Conformer as the encoder and the reworked Transformer as the decoder.
conformer_ctc2 also integrates with the idea of the 'averaging models' in pruned_transducer_stateless4.

The WERs on comparisons with a baseline model, for the librispeech test dataset, are listed as below.

The baseline model is the original conformer CTC attention model trained with icefall/egs/librispeech/ASR/conformer_ctc.
The model is downloaded from  <https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09>.
This model has 12 layers of Conformer encoder layers and 6 Transformer decoder layers.
Number of model parameters is 109,226,120.
It has been trained with 90 epochs with full Librispeech dataset.

For this reworked CTC attention model, it has 12 layers of reworked Conformer encoder layers and 6 reworked Transformer decoder layers.
Number of model parameters is 103,071,035.
With full Librispeech data set, it was trained for **only** 30 epochs because the reworked model would converge much faster.
Please refer to <https://tensorboard.dev/experiment/GR1s6VrJRTW5rtB50jakew/#scalars> to see the loss convergence curve.
Please find the above trained model at <https://huggingface.co/WayneWiser/icefall-asr-librispeech-conformer-ctc2-jit-bpe-500-2022-07-21> in huggingface.

The decoding configuration for the reworked model is --epoch 30, --avg 8, --use-averaged-model True, which is the best after searching.

| WER | reworked ctc attention | with --epoch 30 --avg 8 --use-averaged-model True | | ctc attention| with --epoch 77 --avg 55 | |
|------------------------|-------|------|------|------|------|-----|
| test sets | test-clean | test-other | Avg | test-clean | test-other | Avg |
| ctc-greedy-search      | 2.98% | 7.14%| 5.06%| 2.90%| 7.47%| 5.19%|
| ctc-decoding           | 2.98% | 7.14%| 5.06%| 2.90%| 7.47%| 5.19%|
| 1best                  | 2.93% | 6.37%| 4.65%| 2.70%| 6.49%| 4.60%|
| nbest                  | 2.94% | 6.39%| 4.67%| 2.70%| 6.48%| 4.59%|
| nbest-rescoring        | 2.68% | 5.77%| 4.23%| 2.55%| 6.07%| 4.31%|
| whole-lattice-rescoring| 2.66% | 5.76%| 4.21%| 2.56%| 6.04%| 4.30%|
| attention-decoder      | 2.59% | 5.54%| 4.07%| 2.41%| 5.77%| 4.09%|
| nbest-oracle           | 1.53% | 3.47%| 2.50%| 1.69%| 4.02%| 2.86%|
| rnn-lm                 | 2.37% | 4.98%| 3.68%| 2.31%| 5.35%| 3.83%|



conformer_ctc2 also implements the CTC greedy search decoding, it has the identical WERs with the CTC-decoding method.
For other decoding methods, the average WER of the two test sets with the two models is similar.
Except for the 1best and nbest methods, the overall performance of reworked model is better than the baseline model.


To reproduce the above result, use the following commands.

The training commands are:

```bash
    WORLD_SIZE=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    ./conformer_ctc2/train.py \
    --manifest-dir data/fbank \
    --exp-dir conformer_ctc2/exp \
    --full-libri 1 \
    --spec-aug-time-warp-factor 80 \
    --max-duration 300 \
    --world-size ${WORLD_SIZE} \
    --start-epoch 1 \
    --num-epochs 30 \
    --att-rate 0.7 \
    --num-decoder-layers 6
```


And the following commands are for decoding:

```bash


for method in ctc-greedy-search ctc-decoding 1best nbest-oracle; do
  python3 ./conformer_ctc2/decode.py \
  --exp-dir conformer_ctc2/exp \
  --use-averaged-model True --epoch 30 --avg 8 --max-duration 200 --method $method
done

for method in nbest nbest-rescoring whole-lattice-rescoring attention-decoder ; do
  python3 ./conformer_ctc2/decode.py \
  --exp-dir conformer_ctc2/exp \
  --use-averaged-model True --epoch 30 --avg 8 --max-duration 20 --method $method
done

rnn_dir=$(git rev-parse --show-toplevel)/icefall/rnn_lm
./conformer_ctc2/decode.py \
  --exp-dir conformer_ctc2/exp \
  --lang-dir data/lang_bpe_500 \
  --lm-dir data/lm \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --use-averaged-model True \
  --epoch 30 \
  --avg 8 \
  --nbest-scale 0.5 \
  --rnn-lm-exp-dir ${rnn_dir}/exp \
  --rnn-lm-epoch 29 \
  --rnn-lm-avg 3 \
  --rnn-lm-embedding-dim 2048 \
  --rnn-lm-hidden-dim 2048 \
  --rnn-lm-num-layers 3 \
  --rnn-lm-tie-weights true \
  --method rnn-lm
```

You can find the RNN-LM pre-trained model at
<https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>


### LibriSpeech BPE training results (Conformer-CTC)

#### 2021-11-09

The best WER, as of 2022-06-20, for the librispeech test dataset is below
(using HLG decoding + n-gram LM rescoring + attention decoder rescoring + rnn lm rescoring):

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.32       | 5.39       |

Scale values used in n-gram LM rescoring and attention rescoring for the best WERs are:

| ngram_lm_scale | attention_scale | rnn_lm_scale |
|----------------|-----------------|--------------|
| 0.3            | 2.1             | 2.2          |


To reproduce the above result, use the following commands for training:

```
cd egs/librispeech/ASR/conformer_ctc
./prepare.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./conformer_ctc/train.py \
  --exp-dir conformer_ctc/exp_500_att0.8 \
  --lang-dir data/lang_bpe_500 \
  --att-rate 0.8 \
  --full-libri 1 \
  --max-duration 200 \
  --concatenate-cuts 0 \
  --world-size 4 \
  --bucketing-sampler 1 \
  --start-epoch 0 \
  --num-epochs 90
# Note: It trains for 90 epochs, but the best WER is at epoch-77.pt

# Train the RNN-LM
cd icefall
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./rnn_lm/train.py \
  --exp-dir rnn_lm/exp_2048_3_tied \
  --start-epoch 0 \
  --world-size 4 \
  --num-epochs 30 \
  --use-fp16 1 \
  --embedding-dim 2048 \
  --hidden-dim 2048 \
  --num-layers 3 \
  --batch-size 500 \
  --tie-weights true
```

and the following command for decoding

```
rnn_dir=$(git rev-parse --show-toplevel)/icefall/rnn_lm
./conformer_ctc/decode.py \
  --exp-dir conformer_ctc/exp_500_att0.8 \
  --lang-dir data/lang_bpe_500 \
  --max-duration 30 \
  --concatenate-cuts 0 \
  --bucketing-sampler 1 \
  --num-paths 1000 \
  --epoch 77 \
  --avg 55 \
  --nbest-scale 0.5 \
  --rnn-lm-exp-dir ${rnn_dir}/exp_2048_3_tied \
  --rnn-lm-epoch 29 \
  --rnn-lm-avg 3 \
  --rnn-lm-embedding-dim 2048 \
  --rnn-lm-hidden-dim 2048 \
  --rnn-lm-num-layers 3 \
  --rnn-lm-tie-weights true \
  --method rnn-lm
```

You can find the Conformer-CTC pre-trained model by visiting
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-conformer-ctc-jit-bpe-500-2021-11-09>

and the RNN-LM pre-trained model:
<https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm/tree/main>

The tensorboard log for training is available at
<https://tensorboard.dev/experiment/hZDWrZfaSqOMqtW0NEfXKg/#scalars>


#### 2021-08-19
(Wei Kang): Result of https://github.com/k2-fsa/icefall/pull/13

TensorBoard log is available at https://tensorboard.dev/experiment/GnRzq8WWQW62dK4bklXBTg/#scalars

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_librispeech_conformer_ctc

The best decoding results (WER) are listed below, we got this results by averaging models from epoch 15 to 34, and using `attention-decoder` decoder with num_paths equals to 100.

||test-clean|test-other|
|--|--|--|
|WER| 2.57% | 5.94% |

To get more unique paths, we scaled the lattice.scores with 0.5 (see https://github.com/k2-fsa/icefall/pull/10#discussion_r690951662 for more details), we searched the lm_score_scale and attention_score_scale for best results, the scales that produced the WER above are also listed below.

||lm_scale|attention_scale|
|--|--|--|
|test-clean|1.3|1.2|
|test-other|1.2|1.1|

You can use the following commands to reproduce our results:

```bash
git clone https://github.com/k2-fsa/icefall
cd icefall

# It was using ef233486, you may not need to switch to it
# git checkout ef233486

cd egs/librispeech/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python conformer_ctc/train.py --bucketing-sampler True \
                              --concatenate-cuts False \
                              --max-duration 200 \
                              --full-libri True \
                              --world-size 4 \
                              --lang-dir data/lang_bpe_5000

python conformer_ctc/decode.py --nbest-scale 0.5 \
                               --epoch 34 \
                               --avg 20 \
                               --method attention-decoder \
                               --max-duration 20 \
                               --num-paths 100 \
                               --lang-dir data/lang_bpe_5000
```

### LibriSpeech training results (Tdnn-Lstm)
#### 2021-08-24

(Wei Kang): Result of phone based Tdnn-Lstm model.

Icefall version: https://github.com/k2-fsa/icefall/commit/caa0b9e9425af27e0c6211048acb55a76ed5d315

Pretrained model is available at https://huggingface.co/pkufool/icefall_asr_librispeech_tdnn-lstm_ctc

The best decoding results (WER) are listed below, we got this results by averaging models from epoch 19 to 14, and using `whole-lattice-rescoring` decoding method.

||test-clean|test-other|
|--|--|--|
|WER| 6.59% | 17.69% |

We searched the lm_score_scale for best results, the scales that produced the WER above are also listed below.

||lm_scale|
|--|--|
|test-clean|0.8|
|test-other|0.9|
