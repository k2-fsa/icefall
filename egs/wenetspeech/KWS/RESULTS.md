# Results

## zipformer transducer model

This is a tiny general ASR model, which has around 3.3M parameters, see this PR https://github.com/k2-fsa/icefall/pull/1428 for how to train it and other details.

The modeling units are partial pinyin (i.e initials and finals) with tone.

The positive test sets are from https://github.com/pkufool/open-commands and the negative test set is test net of wenetspeech (has 23 hours audios).

We put the whole pipeline in `run.sh` containing training, decoding and finetuning commands.

The models have been upload to [github](https://github.com/pkufool/keyword-spotting-models/releases/download/v0.11/icefall-kws-zipformer-wenetspeech-20240219.tar.gz).

Here is the results of a small test set which has 20 commands, we list the results of every commands, for
each metric there are two columns, one for the original model trained on wenetspeech L subset, the other
for the finetune model finetuned on in house commands dataset (has 90 hours audio).

> You can see that the performance of the original model is very poor, I think the reason is the test commands are all collected from real product scenarios which are very different from the scenarios wenetspeech dataset was collected. After finetuning, the performance improves a lot.

Commands | FN in positive set | FN in positive set | Recall | Recall | FP in negative set | FP in negative set | False alarm (time / hour)23 hours | False alarm (time / hour)23 hours
-- | -- | -- | -- | -- | -- | -- | -- | --
  | original | finetune | original | finetune | original | finetune | original | finetune
All | 426 / 985 | 40/985 | 56.8% | 95.9% | 7 | 1 | 0.3 | 0.04
下一个 | 5/50 | 0/50 | 90% | 100% | 3 | 0 | 0.13 | 0
开灯 | 19/49 | 2/49 | 61.2% | 95.9% | 0 | 0 | 0 | 0
第一个 | 11/50 | 3/50 | 78% | 94% | 3 | 0 | 0.13 | 0
声音调到最大 | 39/50 | 7/50 | 22% | 86% | 0 | 0 | 0 | 0
暂停音乐 | 36/49 | 1/49 | 26.5% | 98% | 0 | 0 | 0 | 0
暂停播放 | 33/49 | 2/49 | 32.7% | 95.9% | 0 | 0 | 0 | 0
打开卧室灯 | 33/49 | 1/49 | 32.7% | 98% | 0 | 0 | 0 | 0
关闭所有灯 | 27/50 | 0/50 | 46% | 100% | 0 | 0 | 0 | 0
关灯 | 25/48 | 2/48 | 47.9% | 95.8% | 1 | 1 | 0.04 | 0.04
关闭导航 | 25/48 | 1/48 | 47.9% | 97.9% | 0 | 0 | 0 | 0
打开蓝牙 | 24/47 | 0/47 | 48.9% | 100% | 0 | 0 | 0 | 0
下一首歌 | 21/50 | 1/50 | 58% | 98% | 0 | 0 | 0 | 0
换一首歌 | 19/50 | 5/50 | 62% | 90% | 0 | 0 | 0 | 0
继续播放 | 19/50 | 2/50 | 62% | 96% | 0 | 0 | 0 | 0
打开闹钟 | 18/49 | 2/49 | 63.3% | 95.9% | 0 | 0 | 0 | 0
打开音乐 | 17/49 | 0/49 | 65.3% | 100% | 0 | 0 | 0 | 0
打开导航 | 17/48 | 0/49 | 64.6% | 100% | 0 | 0 | 0 | 0
打开电视 | 15/50 | 0/49 | 70% | 100% | 0 | 0 | 0 | 0
大点声 | 12/50 | 5/50 | 76% | 90% | 0 | 0 | 0 | 0
小点声 | 11/50 | 6/50 | 78% | 88% | 0 | 0 | 0 | 0


This is the result of large test set, it has more than 100 commands, too many to list the details of each commands, so only an overall result here. We also list the results of two weak up words 小云小云 (only test set）and 你好问问 (both training and test sets).  For 你好问问, we have to finetune models, one is finetuned on 你好问问 and our in house commands data, the other finetuned on only 你好问问. Both models perform much better than original model, the one finetuned on only 你好问问 behaves slightly better than the other.

> 小云小云 test set and 你好问问 training, dev and test sets are available at https://github.com/pkufool/open-commands

Commands | FN in positive set | FN in positive set | Recall | Recall | FP in negative set | FP in negative set | False alarm (time / hour)23 hours | False alarm (time / hour)23 hours
-- | -- | -- | -- | -- | -- | -- | -- | --
  | original | finetune | original | finetune | original | finetune | original | finetune
large | 2429/4505 | 477 / 4505 | 46.1% | 89.4% | 50 | 41 | 2.17 | 1.78
小云小云（clean) | 30/100 | 40/100 | 70% | 60% | 0 | 0 | 0 | 0
小云小云（noisy) | 118/350 | 154/350 | 66.3% | 56% | 0 | 0 | 0 | 0
你好问问（finetune with all keywords data) | 2236/10641 | 678/10641 | 79% | 93.6% | 0 | 0 | 0 | 0
你好问问(finetune with only 你好问问） | 2236/10641 | 249/10641 | 79% | 97.7% | 0 | 0 | 0 | 0
