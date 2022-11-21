# AMI

This is an ASR recipe for the AMI corpus. AMI provides recordings from the speaker's
headset and lapel microphones, and also 2 array microphones containing 8 channels each.
We pool data in the following 4 ways and train a single model on the pooled data:

(i) individual headset microphone (IHM)
(ii) IHM with simulated reverb
(iii) Single distant microphone (SDM)
(iv) GSS-enhanced array microphones

Speed perturbation and MUSAN noise augmentation are additionally performed on the pooled
data. Here are the statistics of the combined training data:

```python
>>> cuts_train.describe()
Cuts count: 1222053
Total duration (hh:mm:ss): 905:00:28
Speech duration (hh:mm:ss): 905:00:28 (99.9%)
Duration statistics (seconds):
mean    2.7
std     2.8
min     0.0
25%     0.6
50%     1.6
75%     3.8
99%     12.3
99.5%   13.9
99.9%   18.4
max     36.8
```

**Note:** This recipe additionally uses [GSS](https://github.com/desh2608/gss) for enhancement
of far-field array microphones, but this is optional (see `prepare.sh` for details).

## Performance Record

### pruned_transducer_stateless7

The following are decoded using `modified_beam_search`:

| Evaluation set           | dev WER    | test WER |
|--------------------------|------------|---------|
| IHM                      |  18.92  | 17.40 |
| SDM                      |  31.25  | 32.21 |
| MDM (GSS-enhanced)       |  21.67  | 22.43 |

See [RESULTS](/egs/ami/ASR/RESULTS.md) for details.
