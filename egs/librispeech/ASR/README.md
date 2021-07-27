
Run `./prepare.sh` to prepare the data.

Run `./xxx_train.py` (to be added) to train a model.

## Conformer-CTC
Results of the pre-trained model from
`<https://huggingface.co/GuoLiyong/snowfall_bpe_model/tree/main/exp-duration-200-feat_batchnorm-bpe-lrfactor5.0-conformer-512-8-noam>`
are given below

### HLG - no LM rescoring

(output beam size is 8)

#### 1-best decoding

```
[test-clean-no_rescore] %WER 3.15% [1656 / 52576, 127 ins, 377 del, 1152 sub ]
[test-other-no_rescore] %WER 7.03% [3682 / 52343, 220 ins, 1024 del, 2438 sub ]
```

#### n-best decoding

For n=100,

```
[test-clean-no_rescore-100] %WER 3.15% [1656 / 52576, 127 ins, 377 del, 1152 sub ]
[test-other-no_rescore-100] %WER 7.14% [3737 / 52343, 275 ins, 1020 del, 2442 sub ]
```

For n=200,

```
[test-clean-no_rescore-200] %WER 3.16% [1660 / 52576, 125 ins, 378 del, 1157 sub ]
[test-other-no_rescore-200] %WER 7.04% [3684 / 52343, 228 ins, 1012 del, 2444 sub ]
```

### HLG - with LM rescoring

#### Whole lattice rescoring

```
[test-clean-lm_scale_0.8] %WER 2.77% [1456 / 52576, 150 ins, 210 del, 1096 sub ]
[test-other-lm_scale_0.8] %WER 6.23% [3262 / 52343, 246 ins, 635 del, 2381 sub ]
```

WERs of different LM scales are:

```
For test-clean, WER of different settings are:
lm_scale_0.8    2.77    best for test-clean
lm_scale_0.9    2.87
lm_scale_1.0    3.06
lm_scale_1.1    3.34
lm_scale_1.2    3.71
lm_scale_1.3    4.18
lm_scale_1.4    4.8
lm_scale_1.5    5.48
lm_scale_1.6    6.08
lm_scale_1.7    6.79
lm_scale_1.8    7.49
lm_scale_1.9    8.14
lm_scale_2.0    8.82

For test-other, WER of different settings are:
lm_scale_0.8    6.23    best for test-other
lm_scale_0.9    6.37
lm_scale_1.0    6.62
lm_scale_1.1    6.99
lm_scale_1.2    7.46
lm_scale_1.3    8.13
lm_scale_1.4    8.84
lm_scale_1.5    9.61
lm_scale_1.6    10.32
lm_scale_1.7    11.17
lm_scale_1.8    12.12
lm_scale_1.9    12.93
lm_scale_2.0    13.77
```

#### n-best LM rescoring

n = 100

```
[test-clean-lm_scale_0.8] %WER 2.79% [1469 / 52576, 149 ins, 212 del, 1108 sub ]
[test-other-lm_scale_0.8] %WER 6.36% [3329 / 52343, 259 ins, 666 del, 2404 sub ]
```

WERs of different LM scales are:

```
For test-clean, WER of different settings are:
lm_scale_0.8    2.79    best for test-clean
lm_scale_0.9    2.89
lm_scale_1.0    3.03
lm_scale_1.1    3.28
lm_scale_1.2    3.52
lm_scale_1.3    3.78
lm_scale_1.4    4.04
lm_scale_1.5    4.24
lm_scale_1.6    4.45
lm_scale_1.7    4.58
lm_scale_1.8    4.7
lm_scale_1.9    4.8
lm_scale_2.0    4.92
For test-other, WER of different settings are:
lm_scale_0.8    6.36    best for test-other
lm_scale_0.9    6.45
lm_scale_1.0    6.64
lm_scale_1.1    6.92
lm_scale_1.2    7.25
lm_scale_1.3    7.59
lm_scale_1.4    7.88
lm_scale_1.5    8.13
lm_scale_1.6    8.36
lm_scale_1.7    8.54
lm_scale_1.8    8.71
lm_scale_1.9    8.88
lm_scale_2.0    9.02
```
