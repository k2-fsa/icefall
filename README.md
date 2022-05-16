<div align="center">
<img src="https://raw.githubusercontent.com/k2-fsa/icefall/master/docs/source/_static/logo.png" width=168>
</div>

## Installation

Please refer to <https://icefall.readthedocs.io/en/latest/installation/index.html>
for installation.

## Recipes

Please refer to <https://icefall.readthedocs.io/en/latest/recipes/index.html>
for more information.

We provide 6 recipes at present:

  - [yesno][yesno]
  - [LibriSpeech][librispeech]
  - [Aishell][aishell]
  - [TIMIT][timit]
  - [TED-LIUM3][tedlium3]
  - [GigaSpeech][gigaspeech]
  - [Aidatatang_200zh][aidatatang_200zh]

### yesno

This is the simplest ASR recipe in `icefall` and can be run on CPU.
Training takes less than 30 seconds and gives you the following WER:

```
[test_set] %WER 0.42% [1 / 240, 0 ins, 1 del, 0 sub ]
```
We do provide a Colab notebook for this recipe.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tIjjzaJc3IvGyKiMCDWO-TSnBgkcuN3B?usp=sharing)


### LibriSpeech

Please see <https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>
for the **latest** results.

We provide 4 models for this recipe:

- [conformer CTC model][LibriSpeech_conformer_ctc]
- [TDNN LSTM CTC model][LibriSpeech_tdnn_lstm_ctc]
- [Transducer: Conformer encoder + LSTM decoder][LibriSpeech_transducer]
- [Transducer: Conformer encoder + Embedding decoder][LibriSpeech_transducer_stateless]

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

We provide a Colab notebook to run a pre-trained TDNN LSTM CTC model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kNmDXNMwREi0rZGAOIAOJo93REBuOTcd?usp=sharing)


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

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.57       | 5.95       |

#### k2 pruned RNN-T + GigaSpeech

|     | test-clean | test-other |
|-----|------------|------------|
| WER | 2.00       | 4.63       |


### Aishell

We provide two models for this recipe: [conformer CTC model][Aishell_conformer_ctc]
and [TDNN LSTM CTC model][Aishell_tdnn_lstm_ctc].

#### Conformer CTC Model

The best CER we currently have is:

|     | test |
|-----|------|
| CER | 4.26 |


We provide a Colab notebook to run a pre-trained conformer CTC model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WnG17io5HEZ0Gn_cnh_VzK5QYOoiiklC?usp=sharing)

#### Transducer Stateless Model

The best CER we currently have is:

|     | test |
|-----|------|
| CER | 4.68 |


We provide a Colab notebook to run a pre-trained TransducerStateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14XaT2MhnBkK-3_RqqWq3K90Xlbin-GZC?usp=sharing)

#### TDNN LSTM CTC Model

The CER for this model is:

|     | test  |
|-----|-------|
| CER | 10.16 |

We provide a Colab notebook to run a pre-trained TDNN LSTM CTC model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qULaGvXq7PCu_P61oubfz9b53JzY4H3z?usp=sharing)

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

We provide a Colab notebook to run a pre-trained TDNN LiGRU CTC model:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11IT-k4HQIgQngXz1uvWsEYktjqQt7Tmb?usp=sharing)

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

### Aidatatang_200zh

We provide one model for this recipe: [Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss][Aidatatang_200zh
_pruned_transducer_stateless2].

#### Pruned stateless RNN-T: Conformer encoder + Embedding decoder + k2 pruned RNN-T loss

|                      |  Dev  | Test  |
|----------------------|-------|-------|
|    greedy search     | 5.53  | 6.59  |
|   fast beam search   | 5.29  | 6.33  |
| modified beam search | 5.28  | 6.32  |

We provide a Colab notebook to run a pre-trained Pruned Transducer Stateless model: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)(https://colab.research.google.com/drive/1wNSnSj3T5oOctbh5IGCa393gKOoQw2GH?usp=sharing)

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
[Aishell_tdnn_lstm_ctc]: egs/aishell/ASR/tdnn_lstm_ctc
[Aishell_conformer_ctc]: egs/aishell/ASR/conformer_ctc
[TIMIT_tdnn_lstm_ctc]: egs/timit/ASR/tdnn_lstm_ctc
[TIMIT_tdnn_ligru_ctc]: egs/timit/ASR/tdnn_ligru_ctc
[TED-LIUM3_transducer_stateless]: egs/tedlium3/ASR/transducer_stateless
[TED-LIUM3_pruned_transducer_stateless]: egs/tedlium3/ASR/pruned_transducer_stateless
[GigaSpeech_conformer_ctc]: egs/gigaspeech/ASR/conformer_ctc
[GigaSpeech_pruned_transducer_stateless2]: egs/gigaspeech/ASR/pruned_transducer_stateless2
[Aidatatang_200zh_pruned_transducer_stateless2]: egs/aidatatang_200zh/ASR/pruned_transducer_stateless2
[yesno]: egs/yesno/ASR
[librispeech]: egs/librispeech/ASR
[aishell]: egs/aishell/ASR
[timit]: egs/timit/ASR
[tedlium3]: egs/tedlium3/ASR
[gigaspeech]: egs/gigaspeech/ASR
[aidatatang_200zh]: egs/aidatatang_200zh/ASR
[k2]: https://github.com/k2-fsa/k2
