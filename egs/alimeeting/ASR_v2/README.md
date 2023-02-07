
# Introduction

This recipe trains multi-domain ASR models for AliMeeting. By multi-domain, we mean that
we train a single model on close-talk and far-field conditions. This recipe optionally
uses [GSS]-based enhancement for far-field array microphone.
We pool data in the following 4 ways and train a single model on the pooled data:

(i) individual headset microphone (IHM)
(ii) IHM with simulated reverb
(iii) Single distant microphone (SDM)
(iv) GSS-enhanced array microphones

This is different from `alimeeting/ASR` since that recipe trains a model only on the
far-field audio. Additionally, we use text normalization here similar to the original
M2MeT challenge, so the results should be more comparable to those from Table 4 of
the [paper](https://arxiv.org/abs/2110.07393).

The following additional packages need to be installed to run this recipe:
* `pip install jieba`
* `pip install paddlepaddle`
* `pip install git+https://github.com/desh2608/gss.git`

[./RESULTS.md](./RESULTS.md) contains the latest results.

## Performance Record

### pruned_transducer_stateless7

The following are decoded using `modified_beam_search`:

| Evaluation set           | eval WER    | test WER |
|--------------------------|------------|---------|
| IHM                      |  9.58  | 11.53 |
| SDM                      |  23.37  | 25.85 |
| MDM (GSS-enhanced)       |  11.82  | 14.22 |

See [RESULTS](/egs/alimeeting/ASR_v2/RESULTS.md) for details.
