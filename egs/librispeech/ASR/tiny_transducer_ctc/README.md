## Introduction

This recipe is intended for streaming ASR on very low cost devices, with model parameters in the range of 1-2M. It uses a small convolutional net as the encoder. It supports CTC and transducer decoding, and supports both phone and BPE lexicons. For transducer model with phone lexicon, only decoding methods with LG can be used. 

The encoder consists of a subsampling layer followed by a stack of Conv1d-batchnorm-activation-causal_squeeze_excitation blocks, with optional skip add. To reduce latency (at the cost of slightly higher WER), half of the blocks uses causal convolution.

A few remarks:

1. Phone lexicon works better than BPE for CTC decoding (with HLG) but worse for transducer decoding. A possible explanation is that when the decoder failed to output the correct phone it can't be remedied by the external LM.

2. SpecAugment is not helpful for very small models as they tend to underfit.

3. Squeeze-and-excitation module worked like a charm! To make it causal I changed the global average pooling layer to a moving average filter, so only historical context is used.

You can find pretrained models, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02>


## Small encoder

The small encoder uses 10 layers of 1D convolution block with 256 channels, no skip connections. Algorithmic latency is 280ms. Multiply-add ops for the encoder is 22.0Mops.

### CTC decoding with phone lexicon
Total parameters: 1073392
Parameters for CTC decoding: 865816

|                 | test-clean | test-other | comment              |
|-----------------|------------|------------|----------------------|
| 1best           | 9.68       | 24.9       | --epoch 30 --avg 2   |
| nbest-rescoring | 8.2        | 22.7       | --epoch 30 --avg 2   |

### Transducer decoding with BPE 500 lexicon
Total parameters: 1623264
Parameters for transducer decoding: 1237764

|                    | test-clean | test-other | comment              |
|--------------------|------------|------------|----------------------|
| greedy_search      | 14.47      | 32.03      | --epoch 30 --avg 1   |
| fast_beam_search   | 13.38      | 29.61      | --epoch 30 --avg 1   |
|modified_beam_search| 13.02      | 29.32      | --epoch 30 --avg 1   |


## Middle encoder

The middle encoder uses 18 layers of 1D convolution block with 300 channels, with skip connections. Algorithmic latency is 440ms. Multiply-add ops for the encoder is 50.1Mops.

### CTC decoding with phone lexicon
Total parameters: 1073392
Parameters for CTC decoding: 865816

|                 | test-clean | test-other | comment              |
|-----------------|------------|------------|----------------------|
| 1best           | 9.68       | 24.9       | --epoch 30 --avg 2   |
| nbest-rescoring | 8.2        | 22.7       | --epoch 30 --avg 2   |

### Transducer decoding with BPE 500 lexicon
Total parameters: 2735794
Parameters for transducer decoding: 2350294

|                    | test-clean | test-other | comment              |
|--------------------|------------|------------|----------------------|
| greedy_search      | 10.26      | 25.13      | --epoch 30 --avg 2   |
| fast_beam_search   | 9.69       | 23.58      | --epoch 30 --avg 2   |
|modified_beam_search| 9.43       | 23.53      | --epoch 30 --avg 2   |
