# Results

## zipformer transducer model

This is a tiny general ASR model, which has around 3.3M parameters, see this PR https://github.com/k2-fsa/icefall/pull/1428 for how to train it and other details.

The modeling units are 500 BPEs trained on gigaspeech transcripts.

The positive test sets are from https://github.com/pkufool/open-commands and the negative test set is test set of gigaspeech (has 40 hours audios).

We put the whole pipeline in `run.sh` containing training, decoding and finetuning commands.

The models have been upload to [github](https://github.com/pkufool/keyword-spotting-models/releases/download/v0.11/icefall-kws-zipformer-gigaspeech-20240219.tar.gz).

Here is the results of a small test set which has 20 commands, we list the results of every commands, for
each metric there are two columns, one for the original model trained on gigaspeech XL subset, the other
for the finetune model finetuned on commands dataset.

Commands | FN in positive set |FN in positive set | Recall | Recall  | FP in negative set | FP in negative set| False alarm (time / hour) 40 hours | False alarm (time / hour) 40 hours |
-- | -- | -- | -- | --| -- | -- | -- | --
  | original | finetune | original | finetune | original | finetune | original | finetune
All | 43/307 | 4/307 | 86% | 98.7% | 1 | 24 | 0.025 | 0.6
Lights on | 6/17 | 0/17 | 64.7% | 100% | 1 | 9 | 0.025 | 0.225
Heat up | 5/14 | 1/14 | 64.3% | 92.9% | 0 | 1 | 0 | 0.025
Volume down | 4/18 | 0/18 | 77.8% | 100% | 0 | 2 | 0 | 0.05
Volume max | 4/17 | 0/17 | 76.5% | 100% | 0 | 0 | 0 | 0
Volume mute | 4/16 | 0/16 | 75.0% | 100% | 0 | 0 | 0 | 0
Too quiet | 3/17 | 0/17 | 82.4% | 100% | 0 | 4 | 0 | 0.1
Lights off | 3/17 | 0/17 | 82.4% | 100% | 0 | 2 | 0 | 0.05
Play music | 2/14 | 0/14 | 85.7% | 100% | 0 | 0 | 0 | 0
Bring newspaper | 2/13 | 1/13 | 84.6% | 92.3% | 0 | 0 | 0 | 0
Heat down | 2/16 | 2/16 | 87.5% | 87.5% | 0 | 1 | 0 | 0.025
Volume up | 2/18 | 0/18 | 88.9% | 100% | 0 | 1 | 0 | 0.025
Too loud | 1/13 | 0/13 | 92.3% | 100% | 0 | 0 | 0 | 0
Resume music | 1/14 | 0/14 | 92.9% | 100% | 0 | 0 | 0 | 0
Bring shoes | 1/15 | 0/15 | 93.3% | 100% | 0 | 0 | 0 | 0
Switch language | 1/15 | 0/15 | 93.3% | 100% | 0 | 0 | 0 | 0
Pause music | 1/15 | 0/15 | 93.3% | 100% | 0 | 0 | 0 | 0
Bring socks | 1/12 | 0/12 | 91.7% | 100% | 0 | 0 | 0 | 0
Stop music | 0/15 | 0/15 | 100% | 100% | 0 | 0 | 0 | 0
Turn it up | 0/15 | 0/15 | 100% | 100% | 0 | 3 | 0 | 0.075
Turn it down | 0/16 | 0/16 | 100% | 100% | 0 | 1 | 0 | 0.025

This is the result of large test set, it has more than 200 commands, too many to list the details of each commands, so only an overall result here.

Commands | FN in positive set | FN in positive set | Recall | Recall | FP in negative set | FP in negative set | False alarm (time / hour)23 hours | False alarm (time / hour)23 hours
-- | -- | -- | -- | -- | -- | -- | -- | --
  | original | finetune | original | finetune | original | finetune | original | finetune
All | 622/3994 | 79/ 3994 | 83.6% | 97.9% | 18/19930 | 52/19930 | 0.45 | 1.3
