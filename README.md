<div align="center">
<img src="https://raw.githubusercontent.com/k2-fsa/icefall/master/docs/source/_static/logo.png" width=168>
</div>

## Introduction

icefall contains ASR recipes for various datasets
using <https://github.com/k2-fsa/k2>.

You can use <https://github.com/k2-fsa/sherpa> to deploy models
trained with icefall.

You can try pre-trained models from within your browser without the need
to download or install anything by visiting <https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>
See <https://k2-fsa.github.io/icefall/huggingface/spaces.html> for more details.

## Installation

Please refer to <https://icefall.readthedocs.io/en/latest/installation/index.html>
for installation.

## Recipes

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/index.html>
for more information.

We provide the following recipes:

  - [yesno][yesno]
  - [LibriSpeech][librispeech]
  - [GigaSpeech][gigaspeech]
  - [AMI][ami]
  - [Aishell][aishell]
  - [Aishell2][aishell2]
  - [Aishell4][aishell4]
  - [TIMIT][timit]
  - [TED-LIUM3][tedlium3]
  - [Aidatatang_200zh][aidatatang_200zh]
  - [WenetSpeech][wenetspeech]
  - [Alimeeting][alimeeting]
  - [Switchboard][swbd]
  - [TAL_CSASR][tal_csasr]

### yesno

This is the simplest ASR recipe in `icefall` and can be run on CPU.
Training takes less than 30 seconds and gives you the following WER:

```
[test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
```
We provide a Colab notebook for this recipe: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing)


### LibriSpeech

Please see <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>
for the **latest** results.

We provide 5 models for this recipe:

- [conformer CTC model][LibriSpeech_conformer_ctc]
- [TDNN LSTM CTC model][LibriSpeech_tdnn_lstm_ctc]
- [Transducer: Conformer encoder + LSTM decoder][LibriSpeech_transducer]
- [Transducer: Conformer encoder + Embedding decoder][LibriSpeech_transducer_stateless]
- [Transducer: Zipformer encoder + Embedding decoder][LibriSpeech_zipformer]

#### Conformer CTC Model

The best WER we currently have is:

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.42       | 5.73       |


We provide a Colab notebook to run a pre-trained conformer CTC model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1huyupXAcHsUrKaWfI83iMEJ6J0Nh0213?usp=sharing)

#### TDNN LSTM CTC Model

The WER for this model is:

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 6.59       | 17.69      |

We provide a Colab notebook to run a pre-trained TDNN LSTM CTC model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iSfQMp2So-We_Uu49N4AAcMInB72u9z?usp=sharing)


#### Transducer: Conformer encoder + LSTM decoder

Using Conformer as encoder and LSTM as decoder.

The best WER with greedy search is:

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 3.07       | 7.51       |

We provide a Colab notebook to run a pre-trained RNN-T conformer model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_u6yK9jDkPwG_NLrZMN2XK7Aeq4suMO2?usp=sharing)

#### Transducer: Conformer encoder + Embedding decoder

Using Conformer as encoder. The decoder consists of 1 embedding layer
and 1 convolutional layer.

The best WER using modified beam search with beam size 4 is:

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.56       | 6.27       |

Note: No auxiliary losses are used in the training and no LMs are used
in the decoding.

We provide a Colab notebook to run a pre-trained transducer conformer + stateless decoder model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CO1bXJ-2khDckZIW8zjOPHGSKLHpTDlp?usp=sharing)


#### k2 pruned RNN-T

| Encoder         | Params | test-clean | test-other |
|-----------------|--------|------------|------------|
| zipformer       | 65.5M  | 2.21       | 4.79       |
| zipformer-small | 23.2M  | 2.42       | 5.73       |
| zipformer-large | 148.4M | 2.06       | 4.63       |

Note: No auxiliary losses are used in the training and no LMs are used
in the decoding.

#### k2 pruned RNN-T + GigaSpeech

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 1.78       | 4.08       |

Note: No auxiliary losses are used in the training and no LMs are used
in the decoding.

#### k2 pruned RNN-T + GigaSpeech + CommonVoice

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 1.90       | 3.98       |

Note: No auxiliary losses are used in the training and no LMs are used
in the decoding.


### GigaSpeech

We provide two models for this recipe: [Conformer CTC model][GigaSpeech_conformer_ctc]
and [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][GigaSpeech_pruned_transducer_stateless2].

#### Conformer CTC

|     |  Dev  | Test  |
|-----|-------|-------|
| WER | 10.47 | 10.58 |

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy search     | 10.51 | 10.73 |
|   fast beam search   | 10.50 | 10.69 |
| modified beam search | 10.40 | 10.51 |


### Aishell

We provide three models for this recipe: [conformer CTC model][Aishell_conformer_ctc],
[TDNN LSTM CTC model][Aishell_tdnn_lstm_ctc], and [Transducer Stateless Model][Aishell_pruned_transducer_stateless7],

#### Conformer CTC Model

The best CER we currently have is:

|     | test |
|-----|------|
| CER | 4.26 |

#### TDNN LSTM CTC Model

The CER for this model is:

|     | test  |
|-----|-------|
| CER | 10.16 |

We provide a Colab notebook to run a pre-trained TDNN LSTM CTC model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jbyzYq3ytm6j2nlEt-diQm-6QVWyDDEa?usp=sharing)

#### Transducer Stateless Model

The best CER we currently have is:

|     | test |
|-----|------|
| CER | 4.38 |

We provide a Colab notebook to run a pre-trained TransducerStateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14XaT2MhnBkK-3_RqqWq3K90Xlbin-GZC?usp=sharing)


### Aishell2

We provide one model for this recipe: [Transducer Stateless Model][Aishell2_pruned_transducer_stateless5].

#### Transducer Stateless Model

The best WER we currently have is:

|     |   dev-ios  |  test-ios  |
|-----|------------|------------|
| WER |    5.32    |    5.56    |


### Aishell4

We provide one model for this recipe: [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][Aishell4_pruned_transducer_stateless5].

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss (trained with all subsets)

The best CER we currently have is:

|     |   test     |
|-----|------------|
| CER |   29.08    |


We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z3lkURVv9M7uTiIgf3Np9IntMHEknaks?usp=sharing)


### TIMIT

We provide two models for this recipe: [TDNN LSTM CTC model][TIMIT_tdnn_lstm_ctc]
and [TDNN LiGRU CTC model][TIMIT_tdnn_ligru_ctc].

#### TDNN LSTM CTC Model

The best PER we currently have is:

||TEST|
|--|--|
|PER| 19.71% |

We provide a Colab notebook to run a pre-trained TDNN LSTM CTC model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Hs9DA4V96uapw_30uNp32OMJgkuR5VVd?usp=sharing)

#### TDNN LiGRU CTC Model

The PER for this model is:

||TEST|
|--|--|
|PER| 17.66% |

We provide a Colab notebook to run a pre-trained TDNN LiGRU CTC model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z3lkURVv9M7uTiIgf3Np9IntMHEknaks?usp=sharing)


### TED-LIUM3

We provide two models for this recipe: [Transducer Stateless: Conformer encoder + Embedding decoder][TED-LIUM3_transducer_stateless] and [Pruned Transducer Stateless: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][TED-LIUM3_pruned_transducer_stateless].

#### Transducer Stateless:  Conformer encoder + Embedding decoder

The best WER using modified beam search with beam size 4 is:

|     |  dev  |  test  |
|-----|-------|--------|
| WER |  6.91 |  6.33  |

Note: No auxiliary losses are used in the training and no LMs are used in the decoding.

We provide a Colab notebook to run a pre-trained Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MmY5bBxwvKLNT4A2DJnwiqRXhdchUqPN?usp=sharing)

#### Pruned Transducer Stateless: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss

The best WER using modified beam search with beam size 4 is:

|     |  dev  |  test  |
|-----|-------|--------|
| WER |  6.77 |  6.14  |

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1je_1zGrOkGVVd4WLzgkXRHxl-I27yWtz?usp=sharing)


### Aidatatang_200zh

We provide one model for this recipe: [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][Aidatatang_200zh_pruned_transducer_stateless2].

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy search     | 5.53  | 6.59  |
|   fast beam search   | 5.30  | 6.34  |
| modified beam search | 5.27  | 6.33  |

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wNSnSj3T5oOctbh5IGCa393gKOoQw2GH?usp=sharing)


### WenetSpeech

We provide some models for this recipe: [Pruned stateless RNN-T_2: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][WenetSpeech_pruned_transducer_stateless2] and [Pruned stateless RNN-T_5: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][WenetSpeech_pruned_transducer_stateless5].

#### Pruned stateless RNN-T_2: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss (trained with L subset, offline ASR)

|                      |  Dev  | Test-Net | Test-Meeting |
|----------------------|-------|----------|--------------|
|    greedy search     | 7.80  |  8.75    |  13.49       |
| modified beam search| 7.76  |  8.71    |  13.41       |
|   fast beam search   | 7.94  |  8.74    |  13.80       |

#### Pruned stateless RNN-T_5: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss (trained with L subset)
**Streaming**:
|                      |  Dev  | Test-Net | Test-Meeting |
|----------------------|-------|----------|--------------|
| greedy_search | 8.78 | 10.12 | 16.16 |
| modified_beam_search | 8.53| 9.95 | 15.81 |
| fast_beam_search| 9.01 | 10.47 | 16.28 |

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless2 model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EV4e1CHa1GZgEF-bZgizqI9RyFFehIiN?usp=sharing)

### Alimeeting

We provide one model for this recipe: [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][Alimeeting_pruned_transducer_stateless2].

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss (trained with far subset)

|                      |  Eval  | Test-Net |
|----------------------|--------|----------|
|    greedy search     | 31.77  |  34.66   |
|   fast beam search   | 31.39  |  33.02   |
| modified beam search | 30.38  |  34.25   |

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tKr3f0mL17uO_ljdHGKtR7HOmthYHwJG?usp=sharing)


### TAL_CSASR

We provide one model for this recipe: [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][TAL_CSASR_pruned_transducer_stateless5].

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss

The best results for Chinese CER(%) and English WER(%) respectively (zh: Chinese, en: English):
|decoding-method | dev | dev_zh | dev_en | test | test_zh | test_en |
|--|--|--|--|--|--|--|
|greedy_search| 7.30 | 6.48 | 19.19 |7.39| 6.66 | 19.13|
|modified_beam_search| 7.15 | 6.35 | 18.95 | 7.22| 6.50 | 18.70 |
|fast_beam_search| 7.18 | 6.39| 18.90 |  7.27| 6.55 | 18.77|

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DmIx-NloI1CMU5GdZrlse7TRu4y3Dpf8?usp=sharing)

## Deployment with C++

Once you have trained a model in icefall, you may want to deploy it with C++,
without Python dependencies.

Please refer to the documentation
<https://icefall.readthedocs.io/en/latest/recipes/librispeech/conformer_ctc.html#deployment-with-c>
for how to do this.

We also provide a Colab notebook, showing you how to run a torch scripted model in [k2][k2] with C++.
Please see: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BIGLWzS36isskMXHKcqC9ysN6pspYXs_?usp=sharing)


[LibriSpeech_tdnn_lstm_ctc]: egs/librispeech/ASR/tdnn_lstm_ctc
[LibriSpeech_conformer_ctc]: egs/librispeech/ASR/conformer_ctc
[LibriSpeech_transducer]: egs/librispeech/ASR/transducer
[LibriSpeech_transducer_stateless]: egs/librispeech/ASR/transducer_stateless
[LibriSpeech_zipformer]: egs/librispeech/ASR/zipformer
[Aishell_tdnn_lstm_ctc]: egs/aishell/ASR/tdnn_lstm_ctc
[Aishell_conformer_ctc]: egs/aishell/ASR/conformer_ctc
[Aishell_pruned_transducer_stateless7]: egs/aishell/ASR/pruned_transducer_stateless7_bbpe
[Aishell2_pruned_transducer_stateless5]: egs/aishell2/ASR/pruned_transducer_stateless5
[Aishell4_pruned_transducer_stateless5]: egs/aishell4/ASR/pruned_transducer_stateless5
[TIMIT_tdnn_lstm_ctc]: egs/timit/ASR/tdnn_lstm_ctc
[TIMIT_tdnn_ligru_ctc]: egs/timit/ASR/tdnn_ligru_ctc
[TED-LIUM3_transducer_stateless]: egs/tedlium3/ASR/transducer_stateless
[TED-LIUM3_pruned_transducer_stateless]: egs/tedlium3/ASR/pruned_transducer_stateless
[GigaSpeech_conformer_ctc]: egs/gigaspeech/ASR/conformer_ctc
[GigaSpeech_pruned_transducer_stateless2]: egs/gigaspeech/ASR/pruned_transducer_stateless2
[Aidatatang_200zh_pruned_transducer_stateless2]: egs/aidatatang_200zh/ASR/pruned_transducer_stateless2
[WenetSpeech_pruned_transducer_stateless2]: egs/wenetspeech/ASR/pruned_transducer_stateless2
[WenetSpeech_pruned_transducer_stateless5]: egs/wenetspeech/ASR/pruned_transducer_stateless5
[Alimeeting_pruned_transducer_stateless2]: egs/alimeeting/ASR/pruned_transducer_stateless2
[TAL_CSASR_pruned_transducer_stateless5]: egs/tal_csasr/ASR/pruned_transducer_stateless5
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
