<div align="center">
<img src="https://raw.githubusercontent.com/k2-fsa/icefall/master/docs/source/_static/logo.png" width=168>
</div>

# Introduction

The icefall project contains speech-related recipes for various datasets
using [k2-fsa](https://github.com/k2-fsa/k2) and [lhotse](https://github.com/lhotse-speech/lhotse).

You can use [sherpa](https://github.com/k2-fsa/sherpa), [sherpa-ncnn](https://github.com/k2-fsa/sherpa-ncnn) or [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for deployment with models
in icefall; these frameworks also support models not included in icefall; please refer to respective documents for more details.

You can try pre-trained models from within your browser without the need
to download or install anything by visiting this [huggingface space](https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition).
Please refer to [document](https://k2-fsa.github.io/icefall/huggingface/spaces.html) for more details.

# Installation

Please refer to [document](https://k2-fsa.github.io/icefall/installation/index.html)
for installation.

# Recipes

Please refer to [document](https://k2-fsa.github.io/icefall/recipes/index.html)
for more details.

## ASR: Automatic Speech Recognition

### Supported Datasets
  - [yesno][yesno]
  
  - [Aidatatang_200zh][aidatatang_200zh]
  - [Aishell][aishell]
  - [Aishell2][aishell2]
  - [Aishell4][aishell4]
  - [Alimeeting][alimeeting]
  - [AMI][ami]
  - [CommonVoice][commonvoice]
  - [Corpus of Spontaneous Japanese][csj]
  - [GigaSpeech][gigaspeech]
  - [LibriCSS][libricss]
  - [LibriSpeech][librispeech]
  - [Libriheavy][libriheavy]
  - [Multi-Dialect Broadcast News Arabic Speech Recognition][mgb2]
  - [SPGISpeech][spgispeech]
  - [Switchboard][swbd]
  - [TIMIT][timit]
  - [TED-LIUM3][tedlium3]
  - [TAL_CSASR][tal_csasr]
  - [Voxpopuli][voxpopuli]
  - [XBMU-AMDO31][xbmu-amdo31]
  - [WenetSpeech][wenetspeech]
  
More datasets will be added in the future.

### Supported Models

The [LibriSpeech][librispeech] recipe supports the most comprehensive set of models, you are welcome to try them out.

#### CTC 
  - TDNN LSTM CTC
  - Conformer CTC
  - Zipformer CTC

#### MMI
  - Conformer MMI
  - Zipformer MMI

#### Transducer
  - Conformer-based Encoder
  - LSTM-based Encoder
  - Zipformer-based Encoder
  - LSTM-based Predictor
  - [Stateless Predictor](https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/)

#### Whisper
  - [OpenAi Whisper](https://arxiv.org/abs/2212.04356) (We support fine-tuning on AiShell-1.)

If you are willing to contribute to icefall, please refer to [contributing](https://k2-fsa.github.io/icefall/contributing/index.html) for more details.

We would like to highlight the performance of some of the recipes here.

### [yesno][yesno]

This is the simplest ASR recipe in `icefall` and can be run on CPU.
Training takes less than 30 seconds and gives you the following WER:

```
[test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
```
We provide a Colab notebook for this recipe: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing)


### [LibriSpeech][librispeech]

Please see [RESULTS.md](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md)
for the **latest** results.

#### [Conformer CTC](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conformer_ctc)

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.42       | 5.73       |


We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huyupXAcHsUrKaWfI83iMEJ6J0Nh0213?usp=sharing)

#### [TDNN LSTM CTC](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/tdnn_lstm_ctc)

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 6.59       | 17.69      |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iSfQMp2So-We_Uu49N4AAcMInB72u9z?usp=sharing)


#### [Transducer (Conformer Encoder + LSTM Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/transducer)

|               | test-clean | test-other |
|---------------|------------|------------|
| greedy_search | 3.07       | 7.51       |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_u6yK9jDkPwG_NLrZMN2XK7Aeq4suMO2?usp=sharing)

#### [Transducer (Conformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/transducer)

|                                       | test-clean | test-other |
|---------------------------------------|------------|------------|
| modified_beam_search (`beam_size=4`) | 2.56       | 6.27       |


We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CO1bXJ-2khDckZIW8zjOPHGSKLHpTDlp?usp=sharing)


#### [Transducer (Zipformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/zipformer)

WER (modified_beam_search `beam_size=4` unless further stated) 

1. LibriSpeech-960hr

| Encoder         | Params | test-clean | test-other | epochs  | devices    |
|-----------------|--------|------------|------------|---------|------------|
| Zipformer       | 65.5M  | 2.21       | 4.79       | 50      | 4 32G-V100 |
| Zipformer-small | 23.2M  | 2.42       | 5.73       | 50      | 2 32G-V100 |
| Zipformer-large | 148.4M | 2.06       | 4.63       | 50      | 4 32G-V100 |
| Zipformer-large | 148.4M | 2.00       | 4.38       | 174     | 8 80G-A100 |

2. LibriSpeech-960hr + GigaSpeech

| Encoder         | Params | test-clean | test-other |
|-----------------|--------|------------|------------|
| Zipformer       | 65.5M   | 1.78       | 4.08       |


3. LibriSpeech-960hr + GigaSpeech + CommonVoice

| Encoder         | Params | test-clean | test-other |
|-----------------|--------|------------|------------|
| Zipformer       | 65.5M   | 1.90       | 3.98       |


### [GigaSpeech][gigaspeech]

#### [Conformer CTC](https://github.com/k2-fsa/icefall/tree/master/egs/gigaspeech/ASR/conformer_ctc)

|     |  Dev  | Test  |
|-----|-------|-------|
| WER | 10.47 | 10.58 |

#### [Transducer (pruned_transducer_stateless2)](https://github.com/k2-fsa/icefall/tree/master/egs/gigaspeech/ASR/pruned_transducer_stateless2)

Conformer Encoder + Stateless Predictor + k2 Pruned RNN-T Loss

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy_search     | 10.51 | 10.73 |
|   fast_beam_search   | 10.50 | 10.69 |
| modified_beam_search | 10.40 | 10.51 |

#### [Transducer (Zipformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/gigaspeech/ASR/zipformer)

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy_search     | 10.31 | 10.50 |
|   fast_beam_search   | 10.26 | 10.48 |
| modified_beam_search | 10.25 | 10.38 |


### [Aishell][aishell]

#### [TDNN LSTM CTC](https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR/tdnn_lstm_ctc)

|     | test  |
|-----|-------|
| CER | 10.16 |

We provide a Colab notebook to test the pre-trained model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jbyzYq3ytm6j2nlEt-diQm-6QVWyDDEa?usp=sharing)

#### [Transducer (Conformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR/transducer_stateless)

|     | test |
|-----|------|
| CER | 4.38 |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14XaT2MhnBkK-3_RqqWq3K90Xlbin-GZC?usp=sharing)

#### [Transducer (Zipformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/aishell/ASR/zipformer)

WER (modified_beam_search `beam_size=4`) 

| Encoder         | Params | dev | test | epochs  |
|-----------------|--------|-----|------|---------|
| Zipformer       | 73.4M  | 4.13| 4.40 | 55      |
| Zipformer-small | 30.2M  | 4.40| 4.67 | 55      |
| Zipformer-large | 157.3M | 4.03| 4.28 | 56      |


### [Aishell4][aishell4]

#### [Transducer (pruned_transducer_stateless5)](https://github.com/k2-fsa/icefall/tree/master/egs/aishell4/ASR/pruned_transducer_stateless5)

1 Trained with all subsets: 
|     |   test     |
|-----|------------|
| CER |   29.08    |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z3lkURVv9M7uTiIgf3Np9IntMHEknaks?usp=sharing)


### [TIMIT][timit]

#### [TDNN LSTM CTC](https://github.com/k2-fsa/icefall/tree/master/egs/timit/ASR/tdnn_lstm_ctc)

|   |TEST|
|---|----|
|PER| 19.71% |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hs9DA4V96uapw_30uNp32OMJgkuR5VVd?usp=sharing)

#### [TDNN LiGRU CTC](https://github.com/k2-fsa/icefall/tree/master/egs/timit/ASR/tdnn_ligru_ctc)

|   |TEST|
|---|----|
|PER| 17.66% |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z3lkURVv9M7uTiIgf3Np9IntMHEknaks?usp=sharing)


### [TED-LIUM3][tedlium3]

#### [Transducer (Conformer Encoder + Stateless Predictor)](https://github.com/k2-fsa/icefall/tree/master/egs/tedlium3/ASR/transducer_stateless)

|                                      |  dev  |  test  |
|--------------------------------------|-------|--------|
| modified_beam_search (`beam_size=4`) |  6.91 |  6.33  |


We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MmY5bBxwvKLNT4A2DJnwiqRXhdchUqPN?usp=sharing)

#### [Transducer (pruned_transducer_stateless)](https://github.com/k2-fsa/icefall/tree/master/egs/tedlium3/ASR/pruned_transducer_stateless)

|                                      |  dev  |  test  |
|--------------------------------------|-------|--------|
| modified_beam_search (`beam_size=4`) |  6.77 |  6.14  |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1je_1zGrOkGVVd4WLzgkXRHxl-I27yWtz?usp=sharing)


### [Aidatatang_200zh][aidatatang_200zh]

#### [Transducer (pruned_transducer_stateless2)](https://github.com/k2-fsa/icefall/tree/master/egs/aidatatang_200zh/ASR/pruned_transducer_stateless2)

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy_search     | 5.53  | 6.59  |
|   fast_beam_search   | 5.30  | 6.34  |
| modified_beam_search | 5.27  | 6.33  |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wNSnSj3T5oOctbh5IGCa393gKOoQw2GH?usp=sharing)


### [WenetSpeech][wenetspeech]

#### [Transducer (pruned_transducer_stateless2)](https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless2)

|                      |  Dev  | Test-Net | Test-Meeting |
|----------------------|-------|----------|--------------|
|    greedy_search     | 7.80  |  8.75    |  13.49       |
|   fast_beam_search   | 7.94  |  8.74    |  13.80       |
| modified_beam_search | 7.76  |  8.71    |  13.41       |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EV4e1CHa1GZgEF-bZgizqI9RyFFehIiN?usp=sharing)

#### [Transducer **Streaming** (pruned_transducer_stateless5) ](https://github.com/k2-fsa/icefall/tree/master/egs/wenetspeech/ASR/pruned_transducer_stateless5)

|                      |  Dev  | Test-Net | Test-Meeting |
|----------------------|-------|----------|--------------|
| greedy_search | 8.78 | 10.12 | 16.16 |
| fast_beam_search| 9.01 | 10.47 | 16.28 |
| modified_beam_search | 8.53| 9.95 | 15.81 |


### [Alimeeting][alimeeting]

#### [Transducer (pruned_transducer_stateless2)](https://github.com/k2-fsa/icefall/tree/master/egs/alimeeting/ASR/pruned_transducer_stateless2)

|                      |  Eval  | Test-Net |
|----------------------|--------|----------|
|    greedy_search     | 31.77  |  34.66   |
|   fast_beam_search   | 31.39  |  33.02   |
| modified_beam_search | 30.38  |  34.25   |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tKr3f0mL17uO_ljdHGKtR7HOmthYHwJG?usp=sharing)


### [TAL_CSASR][tal_csasr]


#### [Transducer (pruned_transducer_stateless5)](https://github.com/k2-fsa/icefall/tree/master/egs/tal_csasr/ASR/pruned_transducer_stateless5)

The best results for Chinese CER(%) and English WER(%) respectively (zh: Chinese, en: English):
|decoding-method | dev | dev_zh | dev_en | test | test_zh | test_en |
|--|--|--|--|--|--|--|
|greedy_search| 7.30 | 6.48 | 19.19 |7.39| 6.66 | 19.13|
|fast_beam_search| 7.18 | 6.39| 18.90 |  7.27| 6.55 | 18.77|
|modified_beam_search| 7.15 | 6.35 | 18.95 | 7.22| 6.50 | 18.70 |

We provide a Colab notebook to test the pre-trained model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DmIx-NloI1CMU5GdZrlse7TRu4y3Dpf8?usp=sharing)

## TTS: Text-to-Speech

### Supported Datasets

  - [LJSpeech][ljspeech]
  - [VCTK][vctk]
  - [LibriTTS][libritts_tts]

### Supported Models

  - [VITS](https://arxiv.org/abs/2106.06103)

# Deployment with C++

Once you have trained a model in icefall, you may want to deploy it with C++ without Python dependencies.

Please refer to

  - https://k2-fsa.github.io/icefall/model-export/export-with-torch-jit-script.html
  - https://k2-fsa.github.io/icefall/model-export/export-onnx.html
  - https://k2-fsa.github.io/icefall/model-export/export-ncnn.html

for how to do this.

We also provide a Colab notebook, showing you how to run a torch scripted model in [k2][k2] with C++.
Please see: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BIGLWzS36isskMXHKcqC9ysN6pspYXs_?usp=sharing)


[yesno]: egs/yesno/ASR
[librispeech]: egs/librispeech/ASR
[aishell]: egs/aishell/ASR
[aishell2]: egs/aishell2/ASR
[aishell4]: egs/aishell4/ASR
[timit]: egs/timit/ASR
[tedlium3]: egs/tedlium3/ASR
[gigaspeech]: egs/gigaspeech/ASR
[aidatatang_200zh]: egs/aidatatang_200zh/ASR
[wenetspeech]: egs/wenetspeech/ASR
[alimeeting]: egs/alimeeting/ASR
[tal_csasr]: egs/tal_csasr/ASR
[ami]: egs/ami
[swbd]: egs/swbd/ASR
[k2]: https://github.com/k2-fsa/k2
[commonvoice]: egs/commonvoice/ASR
[csj]: egs/csj/ASR
[libricss]: egs/libricss/SURT
[libritts_asr]: egs/libritts/ASR
[libriheavy]: egs/libriheavy/ASR
[mgb2]: egs/mgb2/ASR
[spgispeech]: egs/spgispeech/ASR
[voxpopuli]: egs/voxpopuli/ASR
[xbmu-amdo31]: egs/xbmu-amdo31/ASR

[vctk]: egs/vctk/TTS
[ljspeech]: egs/ljspeech/TTS
[libritts_tts]: egs/libritts/TTS
