# Switchboard

The Switchboard-1 Telephone Speech Corpus (LDC97S62) consists of approximately 260 hours of speech and was originally collected by Texas Instruments in 1990-1, under DARPA sponsorship. The first release of the corpus was published by NIST and distributed by the LDC in 1992-3. Since that release, a number of corrections have been made to the data files as presented on the original CD-ROM set and all copies of the first pressing have been distributed.

Switchboard is a collection of about 2,400 two-sided telephone conversations among 543 speakers (302 male, 241 female) from all areas of the United States. A computer-driven robot operator system handled the calls, giving the caller appropriate recorded prompts, selecting and dialing another person (the callee) to take part in a conversation, introducing a topic for discussion and recording the speech from the two subjects into separate channels until the conversation was finished. About 70 topics were provided, of which about 50 were used frequently. Selection of topics and callees was constrained so that: (1) no two speakers would converse together more than once and (2) no one spoke more than once on a given topic.

(The above introduction is from the [LDC Switchboard-1 Release 2 webpage](https://catalog.ldc.upenn.edu/LDC97S62).)


## Performance Record
|                                |  eval2000  |  rt03  |
|--------------------------------|------------|--------|
|         `conformer_ctc`        |    33.37   |  35.06 |

See [RESULTS](/egs/swbd/ASR/RESULTS.md) for details.

## Credit

The training script for `conformer_ctc` comes from the LibriSpeech `conformer_ctc` recipe in icefall.

A lot of the scripts for data processing are from the first-gen Kaldi and the ESPNet project, tailored by myself to incorporate with Lhotse and Icefall.

Some of the scripts for text normalization are from stale pull requests of [Piotr Żelasko](https://github.com/pzelasko) and [Nagendra Goel](https://github.com/ngoel17).

The `sclite_scoring.py` is from the GigaSpeech recipe for post processing and glm-like scoring, which is definitely not an elegant stuff to do.
