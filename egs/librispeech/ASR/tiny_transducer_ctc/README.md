## Introduction

This recipe is intended for streaming ASR on very low cost devices, with model parameters in the range of 1-2M. It uses a small convolutional net as the encoder. It is trained with combined transducer and CTC losses, and supports both phone and BPE lexicons. For phone lexicon, you can do transducer decoding using a method with LG, but the results were bad. 

The encoder consists of 2 subsampling layers followed by a stack of Conv1d-batchnorm-activation-causal_squeeze_excite blocks, with optional skip connections. To reduce latency (at the cost of slightly higher WER), half of the blocks use causal convolution.

A few remarks & observations:

1. Phone lexicon works better than BPE for CTC decoding (with HLG) but worse for transducer decoding. 

2. SpecAugment is not helpful for very small models as they tend to underfit rather than overfit. For the large model, a less aggressive SpecAugment (see asr_datamodule.py) improved the result a little.

3. Squeeze-and-excitation worked like a charm! It reduces WER quite a bit with marginal increase of parameters and MAC ops. To make it causal I changed the global average pooling layer to a moving average filter, so only historical context is used.

## Pretrained models

You can find pretrained models, training logs, decoding logs, and decoding results at:
<https://huggingface.co/wangtiance/tiny_transducer_ctc/tree/main>

## Results on full libri

I tried 3 different sizes of the encoder. The parameters are around 1M, 2M and 4M, respectively. For CTC decoding, whole-lattice-rescoring frequently causes OOM error so the result is not shown.

### Small encoder

The small encoder uses 10 layers of 1D convolution block with 256 channels, without skip connections. The encoder, decoder and joiner dim is 256. Algorithmic latency is 280ms. Multiply-add ops for the encoder is 22.0Mops. It is more applicable for ASR products with limited vocabulary (like a fixed set of phrases or short sentences). 

#### CTC decoding with phone lexicon
Total parameters: 1073392

Parameters for CTC decoding: 865816

|                 | test-clean | test-other | comment              |
|-----------------|------------|------------|----------------------|
| 1best           | 9.68       | 24.9       | --epoch 30 --avg 2   |
| nbest-rescoring | 8.2        | 22.7       | --epoch 30 --avg 2   |

The training commands are:
```bash

./tiny_transducer_ctc/train.py \
  --num-epochs 30 \
  --full-libri 1 \
  --max-duration 600 \
  --exp-dir tiny_transducer_ctc/exp_small_phone \
  --ctc-loss-scale 0.7 \
  --enable-spec-aug 0 \
  --lang-dir lang_phone \
  --encoder-dim 256 \
  --decoder-dim 256 \
  --joiner-dim 256 \
  --conv-layers 10 \
  --channels 256 \
  --skip-add 0 \
```

#### Transducer decoding with BPE 500 lexicon
Total parameters: 1623264

Parameters for transducer decoding: 1237764

|                    | test-clean | test-other | comment              |
|--------------------|------------|------------|----------------------|
| greedy_search      | 14.47      | 32.03      | --epoch 30 --avg 1   |
| fast_beam_search   | 13.38      | 29.61      | --epoch 30 --avg 1   |
|modified_beam_search| 13.02      | 29.32      | --epoch 30 --avg 1   |

The training commands are:
```bash

./tiny_transducer_ctc/train.py \
  --num-epochs 30 \
  --full-libri 1 \
  --max-duration 600 \
  --exp-dir tiny_transducer_ctc/exp_small_bpe \
  --ctc-loss-scale 0.2 \
  --enable-spec-aug 0 \
  --lang-dir lang_bpe_500 \
  --encoder-dim 256 \
  --decoder-dim 256 \
  --joiner-dim 256 \
  --conv-layers 10 \
  --channels 256 \
  --skip-add 0 \
```

### Middle encoder

The middle encoder uses 18 layers of 1D convolution block with 300 channels, with skip connections. The encoder, decoder and joiner dim is 256. Algorithmic latency is 440ms. Multiply-add ops for the encoder is 50.1Mops. Note that the nbest-rescoring result is better than the tdnn_lstm_ctc recipe with whole-lattice-rescoring.

#### CTC decoding with phone lexicon
Total parameters: 2186242

Parameters for CTC decoding: 1978666

|                 | test-clean | test-other | comment              |
|-----------------|------------|------------|----------------------|
| 1best           | 7.48       | 18.94      | --epoch 30 --avg 1   |
| nbest-rescoring | 6.31       | 16.89      | --epoch 30 --avg 1   |

The training commands are:
```bash

./tiny_transducer_ctc/train.py \
  --num-epochs 30 \
  --full-libri 1 \
  --max-duration 600 \
  --exp-dir tiny_transducer_ctc/exp_middle_phone \
  --ctc-loss-scale 0.7 \
  --enable-spec-aug 0 \
  --lang-dir lang_phone \
  --encoder-dim 256 \
  --decoder-dim 256 \
  --joiner-dim 256 \
  --conv-layers 18 \
  --channels 300 \
  --skip-add 1 \
```

#### Transducer decoding with BPE 500 lexicon
Total parameters: 2735794

Parameters for transducer decoding: 2350294

|                    | test-clean | test-other | comment              |
|--------------------|------------|------------|----------------------|
| greedy_search      | 10.26      | 25.13      | --epoch 30 --avg 2   |
| fast_beam_search   | 9.69       | 23.58      | --epoch 30 --avg 2   |
|modified_beam_search| 9.43       | 23.53      | --epoch 30 --avg 2   |

The training commands are:
```bash

./tiny_transducer_ctc/train.py \
  --num-epochs 30 \
  --full-libri 1 \
  --max-duration 600 \
  --exp-dir tiny_transducer_ctc/exp_middle_bpe \
  --ctc-loss-scale 0.2 \
  --enable-spec-aug 0 \
  --lang-dir lang_bpe_500 \
  --encoder-dim 256 \
  --decoder-dim 256 \
  --joiner-dim 256 \
  --conv-layers 18 \
  --channels 300 \
  --skip-add 1 \
```

### Large encoder

The large encoder uses 18 layers of 1D convolution block with 400 channels, with skip connections. The encoder, decoder and joiner dim is 400. Algorithmic latency is 440ms. Multiply-add ops for the encoder is 88.8Mops. It is interesting to see how much the gap is if we simply scale down more complicated models like Zipformer or emformer.


#### Transducer decoding with BPE 500 lexicon
Total parameters: 4821330

Parameters for transducer decoding: 4219830

|                    | test-clean | test-other | comment              |
|--------------------|------------|------------|----------------------|
| greedy_search      | 8.29       | 21.11      | --epoch 30 --avg 1   |
| fast_beam_search   | 7.91       | 20.1       | --epoch 30 --avg 1   |
|modified_beam_search| 7.74       | 19.89      | --epoch 30 --avg 1   |


The training commands are:
```bash

./tiny_transducer_ctc/train.py \
  --num-epochs 30 \
  --full-libri 1 \
  --max-duration 600 \
  --exp-dir tiny_transducer_ctc/exp_large_bpe \
  --ctc-loss-scale 0.2 \
  --enable-spec-aug 1 \
  --lang-dir lang_bpe_500 \
  --encoder-dim 400 \
  --decoder-dim 400 \
  --joiner-dim 400 \
  --conv-layers 18 \
  --channels 400 \
  --skip-add 1 \
```
