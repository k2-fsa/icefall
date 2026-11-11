# Japanese Common Voice KWS results

## Model and data

The evaluated checkpoint is raw `epoch-60.pt`, selected using development
phone error rate (PER). It is a 2.99M-parameter causal Zipformer Transducer
with 44 Japanese phone tokens, 80-bin fbank input, chunk size 16, and 64 frames
of left context. It was trained on Mozilla Common Voice Scripted Speech 25.0
Japanese. The keyword graph uses the 20 fixed phrases in `eval_keywords.txt`.

| Split | Clips after rate filter | Positive phrase labels | Negative audio | Excluded clips |
| --- | ---: | ---: | ---: | ---: |
| Dev | 8,997 | 716 | 10.23 h | 22 |
| Test | 8,939 | 765 | 10.64 h | 80 |

The 20 phrases were fixed from development transcript counts and phone
collision diagnostics before inspecting test predictions. The train, dev, and
test client-ID sets are disjoint. There are 53 normalized test sentences also
present in train (not necessarily spoken by the same person); this follows the
original Common Voice split. Labels are normalized written-substring matches,
not human word-level time annotations. The preparer's phone audit found that
the isolated keyword phone sequence is absent from the full sentence phone
sequence for 31/716 dev and 32/765 test positive labels, mainly from Japanese
devoicing. It found a keyword phone sequence without the matching written word
in 22 dev and 17 test negative clips. These are label limitations, not measured
model errors.

Source TSV SHA-256: dev
`e7bf7a0cb69d0979cd7b7bfda3aa281c0bd699d30b16c1021d1246f5538c8664`, test
`cacd600fb7ab91024beedfa20d907c64d78f3d15804ca562bce38ffc50b095f3`.
The released keyword list and the code in this recipe regenerate the private
evaluation manifests; no Common Voice audio or transcript is included here.

## KWS operating points

Recall is detected phrase labels / all positive phrase labels. False alarms
per hour count detection events on clips with no written target phrase and
divide by their total audio duration. Icefall's keyword search encodes each
full utterance before searching. The graph used keyword score 1.5, acoustic
threshold 0.35, beam 4, and one trailing blank. The table varies only a
post-filter on recorded acoustic probability; it does not rerun the stateful
graph with a different threshold.

| Post-filter | Dev recall | Dev false alarms/h | Test recall | Test false alarms/h |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 480/716 (67.0%) | 43.22 (442 events) | 525/765 (68.6%) | 40.12 (427 events) |
| 0.45 | 452/716 (63.1%) | 32.46 | 506/765 (66.1%) | 29.41 |
| 0.55 | 388/716 (54.2%) | 19.95 | 439/765 (57.4%) | 16.35 |
| 0.65 | 283/716 (39.5%) | 7.82 | 320/765 (41.8%) | 6.30 |
| 0.75 | 137/716 (19.1%) | 2.15 | 169/765 (22.1%) | 1.69 |
| 0.85 | 46/716 (6.4%) | 0.49 | 45/765 (5.9%) | 0.47 |
| 0.95 | 6/716 (0.8%) | 0.00 | 4/765 (0.5%) | 0.00 |

At the original 0.35 setting, the test set has 494 total false positives,
including 427 on fully negative clips. Micro precision is 51.5%. A lower
false-alarm setting loses most true detections. These results do not support a
production wake-word claim.

At the original 0.35 setting, the test results by phrase for both decoders are:

| Phrase | Icefall detected/positive | Icefall negative false alarms | sherpa-onnx detected/positive | sherpa-onnx negative false alarms |
| --- | ---: | ---: | ---: | ---: |
| こんにちは | 16/23 | 7 | 16/23 | 7 |
| ありがとう | 17/30 | 15 | 19/30 | 24 |
| 世界 | 120/142 | 37 | 128/142 | 46 |
| 大学 | 22/40 | 14 | 21/40 | 20 |
| 問題 | 31/38 | 11 | 31/38 | 11 |
| 東京 | 41/50 | 27 | 42/50 | 31 |
| 日本語 | 21/26 | 29 | 23/26 | 30 |
| 英語 | 9/14 | 36 | 10/14 | 59 |
| 音楽 | 5/14 | 18 | 7/14 | 32 |
| 電話 | 17/37 | 23 | 17/37 | 28 |
| 電車 | 8/14 | 9 | 10/14 | 9 |
| 時間 | 59/85 | 58 | 68/85 | 81 |
| 場所 | 10/24 | 13 | 12/24 | 14 |
| 名前 | 25/32 | 38 | 27/32 | 59 |
| 学校 | 20/37 | 7 | 23/37 | 12 |
| 先生 | 18/21 | 24 | 17/21 | 33 |
| 友達 | 25/41 | 10 | 28/41 | 13 |
| 家族 | 11/15 | 11 | 11/15 | 11 |
| 仕事 | 26/44 | 25 | 31/44 | 35 |
| 天気 | 24/38 | 15 | 25/38 | 19 |

sherpa-onnx 1.13.8 consumed 0.16-second chunks with its online
KeywordSpotter and 0.8 seconds of trailing silence. It used the same keyword
score 1.5, graph threshold 0.35, path count 4, and one trailing blank. The
ONNX components came from Icefall's WenetSpeech KWS streaming exporter using
the same epoch-60 weights.

| Decoder | Dev recall | Dev false alarms/h | Test recall | Test false alarms/h | Test micro precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Icefall full-audio, threshold 0.35 | 480/716 (67.0%) | 43.22 | 525/765 (68.6%) | 40.12 | 51.5% |
| sherpa-onnx streaming, threshold 0.35 | 528/716 (73.7%) | 58.66 | 566/765 (74.0%) | 53.93 | 45.5% |

The streaming decoder detects more targets and also produces more false
alarms. Its test rate is 574 detection events in 10.64 negative hours; total
false positives including positive clips are 678. No streaming trigger delay
relative to a word-level ground-truth boundary was measured.

This is a read-speech phrase-spotting test. It does not measure real command
speech, background noise, far-field audio, or trigger latency. Its numbers
must not be directly compared with WenetSpeech KWS, which uses different
keywords, speakers, language, and negative audio.

## Phone recognition sanity check

For this checkpoint, modified beam search with beam 4 and blank penalty 0.5
achieved 21.88% PER on filtered dev, 24.09% PER on filtered test, and 26.49%
PER on the full Common Voice test split. This is a different decoder and metric
from KWS, and is not evidence of a usable false-alarm rate.
