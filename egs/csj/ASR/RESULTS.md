# Results

## Streaming Zipformer-Transducer (Pruned Stateless Transducer + Streaming Zipformer)

### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

See <https://github.com/k2-fsa/icefall/pull/892> for more details.

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208>

Number of model parameters: 75688409, i.e. 75.7M.

#### training on disfluent transcript

The CERs are:

| decoding method	         | chunk size | eval1 | eval2 | eval3 | excluded | valid | average through     | decoding mode       |
| -------------------------- | ---------- | ----- | ----- | ----- | -------- | ----- | ------------------- | ------------------- |
| fast beam search           | 320ms	  | 6.27  | 5.13  | 5.05  | 6.30     | 5.42  | 14->30 | simulated streaming |
| fast beam search           | 320ms	  | 5.91  | 4.30  | 4.53  | 6.13     | 5.13  | 14->30 | chunk-wise          |
| greedy search              | 320ms      | 5.81  | 4.29  | 4.63  | 6.07     | 5.13  | 14->30 | simulated streaming |
| greedy search              | 320ms      | 5.91  | 4.50  | 4.65  | 6.36     | 5.34  | 14->30 | chunk-wise          |
| modified beam search       | 320ms      | 5.59  | 4.20  | 4.39  | 5.54     | 4.90  | 14->30 | simulated streaming |
| modified beam search       | 320ms      | 5.79  | 4.48  | 4.41  | 5.98     | 5.19  | 14->30 | chunk-wise          |
| fast beam search           | 640ms      | 5.76  | 4.35  | 4.39  | 5.40     | 4.92  | 14->30 | simulated streaming |
| fast beam search           | 640ms      | 5.45  | 4.31  | 4.29  | 5.61     | 4.97  | 14->30 | chunk-wise          |
| greedy search              | 640ms      | 5.37  | 3.94  | 4.03  | 5.22     | 4.77  | 14->30 | simulated streaming |
| greedy search              | 640ms      | 5.77  | 4.44  | 4.49  | 5.70     | 5.29  | 14->30 | chunk-wise          |
| modified beam search       | 640ms      | 5.19  | 3.81  | 3.93  | 4.83     | 4.59  | 14->30 | simulated streaming |
| modified beam search       | 640ms      | 6.71  | 5.35  | 4.95  | 6.06     | 5.94  | 14->30 | chunk-wise          |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command was:
```bash
./pruned_transducer_stateless7_streaming/train.py \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --context-size 2 \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent \
  --max-duration 375 \
  --transcript-mode disfluent \
  --lang data/lang_char \
  --musan-dir /mnt/host/corpus/musan/musan/fbank
```

Padding at 30 caused many insertions at the end of utterances. The simulated streaming decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent_2 \
            --epoch 30 \
            --avg 14 \
            --max-duration 250 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode disfluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_disfluent_2/github/sim_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --pad 4
        done
    done
done
```

The streaming chunk-wise decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/streaming_decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent_2 \
            --epoch 30 \
            --avg 14 \
            --max-duration 250 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode disfluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_disfluent_2/github/stream_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --num-decode-streams 40
    done
done
```

#### training on fluent transcript

The CERs are:

| decoding method  |  chunk size  |  eval1  |  eval2  |  eval3  |  excluded  |  valid  |  average through  |  decoding mode |
| ---------------  | ------------ | ------- | ------ | ------- | ----------- | ------- | --------- | -------------- |
| fast beam search pad30  |  320ms  |  4.72  |  3.74  |  4.21  |  5.21  |  4.39  |  19->30  |  simulated streaming |
| fast beam search  |  320ms  |  4.63  |  3.63  |  4.18  |  5.3  |  4.31  |  19->30  |  chunk-wise |
| greedy search  |  320ms  |  4.83  |  3.71  |  4.27  |  4.89  |  4.38  |  19->30  |  simulated streaming |
| greedy search  |  320ms  |  4.7  |  3.87  |  4.24  |  5.39  |  4.39  |  19->30  |  chunk-wise |
| modified beam search  |  320ms  |  4.61  |  3.55  |  4.07  |  4.89  |  4.18  |  19->30  |  simulated streaming |
| modified beam search  |  320ms  |  4.53  |  3.73  |  3.98  |  5.9  |  4.25  |  19->30  |  chunk-wise |
| fast beam search pad30  |  640ms  |  4.33  |  3.55  |  4.03  |  4.97  |  4.33  |  19->30  |  simulated streaming |
| fast beam search  |  640ms  |  4.21  |  3.64  |  3.93  |  5.04  |  4.18  |  19->30  |  chunk-wise |
| greedy search  |  640ms  |  4.3  |  3.51  |  3.91  |  4.45  |  4.04  |  19->30  |  simulated streaming |
| greedy search  |  640ms  |  4.4  |  3.83  |  4.03  |  5.14  |  4.31  |  19->30  |  chunk-wise |
| modified beam search  |  640ms  |  4.11  |  3.29  |  3.66  |  4.33  |  3.88  |  19->30  |  simulated streaming |
| modified beam search  |  640ms  |  4.42  |  3.91  |  3.93  |  5.62  |  4.33  |  19->30  |  chunk-wise |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command was:
```bash
./pruned_transducer_stateless7_streaming/train.py \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --context-size 2 \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2 \
  --max-duration 375 \
  --transcript-mode fluent \
  --telegram-cred misc.ini \
  --lang data/lang_char \
  --manifest-dir $csj_fbank_dir \
  --musan-dir /mnt/host/corpus/musan/musan/fbank
```

The simulated streaming decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2 \
            --epoch 30 \
            --avg 19 \
            --max-duration 350 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode fluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_fluent_2/github/sim_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --pad 4
    done
    # Padding of 4 caused many deletions only in the fast_beam_search case.
    python pruned_transducer_stateless7_streaming/decode.py \
        --feedforward-dims  "1024,1024,2048,2048,1024" \
        --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2 \
        --epoch 30 \
        --avg 19 \
        --max-duration 350 \
        --decoding-method fast_beam_search \
        --manifest-dir /mnt/host/corpus/csj/fbank \
        --lang data/lang_char \
        --transcript-mode fluent \
        --res-dir pruned_transducer_stateless7_streaming/exp_fluent_2/github/sim_"$chunk"_fast_beam_search \
        --decode-chunk-len $chunk \
        --pad 30
done
```

The streaming chunk-wise decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/streaming_decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2 \
            --epoch 30 \
            --avg 19 \
            --max-duration 250 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode fluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_fluent_2/github/stream_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --gpu 4 \
            --num-decode-streams 40
    done
done
```

#### Comparing disfluent to fluent

$$ \texttt{CER}^{f}_d = \frac{\texttt{sub}_f + \texttt{ins} + \texttt{del}_f}{N_f} $$

This comparison evaluates the disfluent model on the fluent transcript (calculated by `disfluent_recogs_to_fluent.py`), forgiving the disfluent model's mistakes on fillers and partial words. It is meant as an illustrative metric only, so that the disfluent and fluent models can be compared.

| decoding method | chunk size | eval1 (d vs f) | eval2 (d vs f) | eval3  (d vs f) | excluded  (d vs f) | valid  (d vs f) | decoding mode |
| --------------- | ---------- | -------------- | -------------- | --------------- | ------------------ | --------------- | ------------- |
| fast beam search | 320ms | 5.44 vs 4.72 | 4.49 vs 3.74 | 4.44 vs 4.21 | 5.14 vs 5.21 | 4.64 vs 4.39 | simulated streaming |
| fast beam search | 320ms | 5.05 vs 4.63 | 3.63 vs 3.63 | 3.91 vs 4.18 | 4.75 vs 5.30 | 4.29 vs 4.31 | chunk-wise |
| greedy search | 320ms | 4.97 vs 4.83 | 3.63 vs 3.71 | 4.02 vs 4.27 | 4.93 vs 4.89 | 4.32 vs 4.38 | simulated streaming |
| greedy search | 320ms | 5.02 vs 4.70 | 3.78 vs 3.87 | 4.02 vs 4.24 | 5.11 vs 5.39 | 4.47 vs 4.39 | chunk-wise  |
| modified beam search | 320ms | 4.86 vs 4.61 | 3.62 vs 3.55 | 3.85 vs 4.07 | 4.66 vs 4.89 | 4.21 vs 4.18 | simulated streaming |
| modified beam search | 320ms | 5.05 vs 4.53 | 3.89 vs 3.73 | 3.88 vs 3.98 | 4.88 vs 5.90 | 4.48 vs 4.25 | chunk-wise |
| fast beam search | 640ms | 4.93 vs 4.33 | 3.74 vs 3.55 | 3.78 vs 4.03 | 4.31 vs 4.97 | 4.15 vs 4.33 | simulated streaming |
| fast beam search | 640ms | 4.61 vs 4.21 | 3.67 vs 3.64 | 3.66 vs 3.93 | 4.34 vs 5.04 | 4.15 vs 4.18 | chunk-wise |
| greedy search | 640ms | 4.48 vs 4.30 | 3.29 vs 3.51 | 3.43 vs 3.91 | 4.11 vs 4.45 | 3.96 vs 4.04 | simulated streaming |
| greedy search | 640ms | 4.89 vs 4.40 | 3.77 vs 3.83 | 3.87 vs 4.03 | 4.41 vs 5.14 | 4.47 vs 4.31 | chunk-wise |
| modified beam search | 640ms | 4.45 vs 4.11 | 3.28 vs 3.29 | 3.41 vs 3.66 | 3.97 vs 4.33 | 3.90 vs 3.88 | simulated streaming |
| modified beam search | 640ms | 6.10 vs 4.42 | 4.86 vs 3.91 | 4.51 vs 3.93 | 5.16 vs 5.62 | 5.34 vs 4.33 | chunk-wise |
| __average of (d - f)__ |  | 0.50 | 0.14 | -0.13 | -0.45 | 0.11 | |
