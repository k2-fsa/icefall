# Introduction

This is a multi-talker ASR recipe for the AMI and ICSI datasets. We train a Streaming
Unmixing and Recognition Transducer (SURT) model for the task.

Please refer to the `egs/libricss/SURT` recipe README for details about the task and the
model.

## Description of the recipe

### Pre-requisites

The recipes in this directory need the following packages to be installed:

- [meeteval](https://github.com/fgnt/meeteval)
- [einops](https://github.com/arogozhnikov/einops)

Additionally, we initialize the model with the pre-trained model from the LibriCSS recipe.
Please download this checkpoint (see below) or train the LibriCSS recipe first.

### Training

To train the model, run the following from within `egs/ami/SURT`:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python dprnn_zipformer/train.py \
    --use-fp16 True \
    --exp-dir dprnn_zipformer/exp/surt_base \
    --world-size 4 \
    --max-duration 500 \
    --max-duration-valid 250 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 30 \
    --enable-spec-aug True \
    --enable-musan False \
    --ctc-loss-scale 0.2 \
    --heat-loss-scale 0.2 \
    --base-lr 0.004 \
    --model-init-ckpt exp/libricss_base.pt \
    --chunk-width-randomization True \
    --num-mask-encoder-layers 4 \
    --num-encoder-layers 2,2,2,2,2
```

The above is for SURT-base (~26M). For SURT-large (~38M), use:

```bash
    --model-init-ckpt exp/libricss_large.pt \
    --num-mask-encoder-layers 6 \
    --num-encoder-layers 2,4,3,2,4 \
    --model-init-ckpt exp/zipformer_large.pt \
```

**NOTE:** You may need to decrease the `--max-duration` for SURT-large to avoid OOM.

### Adaptation

The training step above only trains on simulated mixtures. For best results, we also
adapt the final model on the AMI+ICSI train set. For this, run the following from within
`egs/ami/SURT`:

```bash
export CUDA_VISIBLE_DEVICES="0"

python dprnn_zipformer/train_adapt.py \
    --use-fp16 True \
    --exp-dir dprnn_zipformer/exp/surt_base_adapt \
    --world-size 4 \
    --max-duration 500 \
    --max-duration-valid 250 \
    --max-cuts 200 \
    --num-buckets 50 \
    --num-epochs 8 \
    --lr-epochs 2 \
    --enable-spec-aug True \
    --enable-musan False \
    --ctc-loss-scale 0.2 \
    --base-lr 0.0004 \
    --model-init-ckpt dprnn_zipformer/exp/surt_base/epoch-30.pt \
    --chunk-width-randomization True \
    --num-mask-encoder-layers 4 \
    --num-encoder-layers 2,2,2,2,2
```

For SURT-large, use the following config:

```bash
    --num-mask-encoder-layers 6 \
    --num-encoder-layers 2,4,3,2,4 \
    --model-init-ckpt dprnn_zipformer/exp/surt_large/epoch-30.pt \
    --num-epochs 15 \
    --lr-epochs 4 \
```


### Decoding

To decode the model, run the following from within `egs/ami/SURT`:

#### Greedy search

```bash
export CUDA_VISIBLE_DEVICES="0"

python dprnn_zipformer/decode.py \
    --epoch 20 --avg 1 --use-averaged-model False \
    --exp-dir dprnn_zipformer/exp/surt_base_adapt \
    --max-duration 250 \
    --decoding-method greedy_search
```

#### Beam search

```bash
python dprnn_zipformer/decode.py \
    --epoch 20 --avg 1 --use-averaged-model False \
    --exp-dir dprnn_zipformer/exp/surt_base_adapt \
    --max-duration 250 \
    --decoding-method modified_beam_search \
    --beam-size 4
```

## Results (using beam search)

**AMI**

| Model      | IHM-Mix |  SDM |  MDM |
|------------|:-------:|:----:|:----:|
| SURT-base  |   39.8  | 65.4 | 46.6 |
|   + adapt  |   37.4  | 46.9 | 43.7 |
| SURT-large |   36.8  | 62.5 | 44.4 |
|    + adapt |   **35.1**  | **44.6** | **41.4** |

**ICSI**

| Model      | IHM-Mix |  SDM |
|------------|:-------:|:----:|
| SURT-base  |   28.3  | 60.0 |
|   + adapt  |   26.3  | 33.9 |
| SURT-large |   27.8  | 59.7 |
|    + adapt |   **24.4**  | **32.3** |

## Pre-trained models and logs

* LibriCSS pre-trained model (for initialization): [base](https://huggingface.co/desh2608/icefall-surt-libricss-dprnn-zipformer/tree/main/exp/surt_base) [large](https://huggingface.co/desh2608/icefall-surt-libricss-dprnn-zipformer/tree/main/exp/surt_large)

* Pre-trained models: <https://huggingface.co/desh2608/icefall-surt-ami-dprnn-zipformer>

* Training logs:
    - surt_base: <https://tensorboard.dev/experiment/8awy98VZSWegLmH4l2JWSA/>
    - surt_base_adapt: <https://tensorboard.dev/experiment/aGVgXVzYRDKbGUbPekcNjg/>
    - surt_large: <https://tensorboard.dev/experiment/ZXMkez0VSYKbPLqRk4clOQ/>
    - surt_large_adapt: <https://tensorboard.dev/experiment/WLKL1e7bTVyEjSonYSNYwg/>
