## wer with various right context

related model and decoding result/log fils could be found:
https://huggingface.co/GuoLiyong/icefall_streaming_prunned_transducer_stateless/tree/main/streaming_pruned_transducer_stateless/exp

decoding with ctc greedy search:

right_context|1|8|16|32|64|full
--|--|--|--|--|--|--
latency|0.07s|0.35s|0.67s|1.31s|2.59s|*
test_clean|5.60|4.00|3.76|3.75|3.65|3.28|
+20 tailing dummy frames|5.52|3.98|3.75|3.75|3.65|3.28
simulate streaming with chunk_by_chunk decoding|5.52|3.98|3.75|3.75|3.65|3.28
test_other|14.07|10.69|9.80|9.48|9.01|8.05|
+20 tailing dummy frames|14.00|10.69|9.80|9.48|9.0|8.04
simulate streaming with chunk_by_chunk decoding|14.00|10.69|9.80|9.48|9.0|8.04



## How latency is computed?

latency = (subsampling factor * right_context + initialize_frames_need_by_subsampling_convs) * 10ms

During which: subsmapling factor = 4 
initialize_frames_need_by_subsampling_convs = 3

To decode the first frame encoder out: 7 frams fbanks = subsampling_factor + initialize_frames_need_by_subsampling_convs are needed. 
Once the deocding started, 4 frames fbank are needed per encoder_out frame.


## Why does tailing dummy frames help?

As 4 frames fbank are needed per encoder_out frame, suppose only 3(or 2,1) frames left, after a decoding process. 
There will no encoder out frames corresponding to these 3 frames. 
This may results in some "substitution/deletion errors" at the end. 
By padding some dummy frames to the right, this problem could be solved to some extent.

### Some Examples results:
padding 0 frame|padding 20 frames
--|--
WITH ONE JUMP (ANDERS->ANDREWS) GOT OUT OF HIS (CHAIR->CHA)|WITH ONE JUMP (ANDERS->ANDREWS) GOT OUT OF HIS CHAIR
COME WE'LL HAVE OUR COFFEE IN THE OTHER ROOM AND YOU CAN (SMOKE->SMO)|COME WE'LL HAVE OUR COFFEE IN THE OTHER ROOM AND YOU CAN SMOKE
THINKING OF ALL THIS I WENT TO (SLEEP->SLEE)|THINKING OF ALL THIS I WENT TO SLEEP 
STEAM UP AND CANVAS SPREAD THE SCHOONER STARTED (EASTWARDS->EASTWARD)|STEAM UP AND CANVAS SPREAD THE SCHOONER STARTED EASTWARDS

### final Wers and detail error counts :
*|wer|ins|del|sub
--|--|--|--|--
padding 0|5.60|329|283|2332
padding 20 frames|5.52|329|282|2291

Raw log files of previous table:
```
padding 0 frames:
%WER = 5.60 
Errors: 329 insertions, 283 deletions, 2332 substitutions, over 52576 reference words (49961 correct)

padding 20 frames:
%WER = 5.52
Errors: 329 insertions, 282 deletions, 2291 substitutions, over 52576 reference words (50003 correct) 
```


