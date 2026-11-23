# Faster Zipformer training with fused GPU kernels

`fast_kernels.py` contains opt-in replacements for the most memory-bound
parts of a Zipformer training step. They compute the same values as the
reference code in `scaling.py` (up to floating-point rounding) and only
change training speed. Decoding, export and the saved checkpoints are not
affected.

They are **off by default**. Enable them with the environment variable
`ZIPFORMER_FAST_KERNELS`.

| value | effect |
|---|---|
| unset, `0` | upstream behaviour (default) |
| `1` or `all` | enable every feature below |
| comma list, e.g. `balancer,softmax` | enable only the listed features |

| feature | what it replaces |
|---|---|
| `actlin` | `ActivationDropoutAndLinear`: one Triton kernel for `swoosh(x) * dropout_mask` that writes the matmul dtype directly, plus a fused backward |
| `swoosh` | `SwooshL` / `SwooshR` modules: Triton kernel that keeps the input dtype (k2's op returns float32) |
| `softmax` | `SoftmaxFunction` backward: native fused softmax backward instead of float32 elementwise ops |
| `balancer` | `BalancerFunction` backward: closed-form gradient (two small kernels) instead of re-running a float32 autograd graph |
| `channels_last` | `Conv2dSubsampling` runs in NHWC so cuDNN's fast depthwise kernels are used |

Requirements: a CUDA GPU and Triton (installed with PyTorch on Linux).
If either is missing, a warning is logged and the reference code is used.
Tested with torch 2.9.1 + CUDA 12.6 on an NVIDIA L40S.

## Sample commands

Train with all optimizations: prepend the variable to the usual command.

```bash
cd egs/librispeech/ASR
export CUDA_VISIBLE_DEVICES="0,1,2,3"

ZIPFORMER_FAST_KERNELS=1 ./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 1000
```

Large model: the same variable works for any model size.

```bash
ZIPFORMER_FAST_KERNELS=1 ./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-large \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --max-duration 1000
```

Only some of the optimizations:

```bash
ZIPFORMER_FAST_KERNELS=balancer,softmax ./zipformer/train.py ...
```

Decoding needs no change (no environment variable):

```bash
./zipformer/decode.py --epoch 30 --avg 9 --exp-dir zipformer/exp \
  --max-duration 600 --decoding-method greedy_search
```

Check the kernels against the reference implementation (needs a GPU):

```bash
python3 ./zipformer/test_fast_kernels.py
```

## Results

Zipformer medium (65.5M), LibriSpeech train-clean-100, fp16,
max-duration 1000, one NVIDIA L40S per run, 30 epochs, same seed:

| | upstream | `ZIPFORMER_FAST_KERNELS=1` |
|---|---|---|
| total training time | 7503 s | 5641 s (1.33x faster, -25% GPU time) |
| test-clean / test-other WER, greedy search | 7.53 / 20.20 | 7.43 / 20.31 |
| test-clean / test-other WER, modified beam search | 7.39 / 19.62 | 7.27 / 19.73 |

Synthetic training step, bf16, max-duration 1000: 1577 -> 2161 seconds of
audio per second (+36%).

The dropout random numbers are drawn differently, so losses are not
bit-identical to upstream; WER is the same within run-to-run noise.
