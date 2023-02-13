# Results

## Streaming Zipformer-Transducer (Pruned Stateless Transducer + Streaming Zipformer)

### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

See <https://github.com/k2-fsa/icefall/pull/892> for more details.

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/TeoWenShen/icefall-asr-csj-pruned-transducer-stateless7-streaming-230208>

Number of model parameters: 75688409, i.e. 75.7M.

#### training on disfluent transcript

The CERs are:

| decoding method | chunk size | eval1 | eval2 | eval3 | excluded | valid | average | decoding mode |
| --------------- | ---------- | ----- | ----- | ----- | -------- | ----- | ------- | ------------- |
| fast beam search | 320ms | 5.39 | 4.08 | 4.16 | 5.4 | 5.02 | --epoch 30 --avg 17 | simulated streaming |
| fast beam search | 320ms | 5.34 | 4.1 | 4.26 | 5.61 | 4.91 | --epoch 30 --avg 17 | chunk-wise |
| greedy search | 320ms | 5.43 | 4.14 | 4.31 | 5.48 | 4.88 | --epoch 30 --avg 17 | simulated streaming |
| greedy search | 320ms | 5.44 | 4.14 | 4.39 | 5.7 | 4.98 | --epoch 30 --avg 17 | chunk-wise |
| modified beam search | 320ms | 5.2 | 3.95 | 4.09 | 5.12 | 4.75 | --epoch 30 --avg 17 | simulated streaming |
| modified beam search | 320ms | 5.18 | 4.07 | 4.12 | 5.36 | 4.77 | --epoch 30 --avg 17 | chunk-wise |
| fast beam search | 640ms | 5.01 | 3.78 | 3.96 | 4.85 | 4.6 | --epoch 30 --avg 17 | simulated streaming |
| fast beam search | 640ms | 4.97 | 3.88 | 3.96 | 4.91 | 4.61 | --epoch 30 --avg 17 | chunk-wise |
| greedy search | 640ms | 5.02 | 3.84 | 4.14 | 5.02 | 4.59 | --epoch 30 --avg 17 | simulated streaming |
| greedy search | 640ms | 5.32 | 4.22 | 4.33 | 5.39 | 4.99 | --epoch 30 --avg 17 | chunk-wise |
| modified beam search | 640ms | 4.78 | 3.66 | 3.85 | 4.72 | 4.42 | --epoch 30 --avg 17 | simulated streaming |
| modified beam search | 640ms | 5.77 | 4.72 | 4.73 | 5.85 | 5.36 | --epoch 30 --avg 17 | chunk-wise |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command was:
```bash
./pruned_transducer_stateless7_streaming/train.py \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent_2_pad30 \
  --max-duration 375 \
  --transcript-mode disfluent \
  --lang data/lang_char \
  --manifest-dir /mnt/host/corpus/csj/fbank \
  --pad-feature 30 \
  --musan-dir /mnt/host/corpus/musan/musan/fbank
```

The simulated streaming decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent_2_pad30 \
            --epoch 30 \
            --avg 17 \
            --max-duration 350 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode disfluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_disfluent_2_pad30/github/sim_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --pad-feature 30 \
            --gpu 0
    done
done
```

The streaming chunk-wise decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/streaming_decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_disfluent_2_pad30 \
            --epoch 30 \
            --avg 17 \
            --max-duration 350 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode disfluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_disfluent_2_pad30/github/stream_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --gpu 2 \
            --num-decode-streams 40
    done
done
```

#### training on fluent transcript

The CERs are:

| decoding method | chunk size | eval1 | eval2 | eval3 | excluded | valid | average | decoding mode |
| --------------- | ---------- | ----- | ----- | ----- | -------- | ----- | ------- | ------------- |
| fast beam search | 320ms | 4.19 | 3.63 | 3.77 | 4.43 | 4.09 | --epoch 30 --avg 12 | simulated streaming |
| fast beam search | 320ms | 4.06 | 3.55 | 3.66 | 4.70 | 4.04 | --epoch 30 --avg 12 | chunk-wise |
| greedy search | 320ms | 4.22 | 3.62 | 3.82 | 4.45 | 3.98 | --epoch 30 --avg 12 | simulated streaming |
| greedy search | 320ms | 4.13 | 3.61 | 3.85 | 4.67 | 4.05 | --epoch 30 --avg 12 | chunk-wise |
| modified beam search | 320ms | 4.02 | 3.43 | 3.62 | 4.43 | 3.81 | --epoch 30 --avg 12 | simulated streaming |
| modified beam search | 320ms | 3.97 | 3.43 | 3.59 | 4.99 | 3.88 | --epoch 30 --avg 12 | chunk-wise |
| fast beam search | 640ms | 3.80 | 3.31 | 3.55 | 4.16 | 3.90 | --epoch 30 --avg 12 | simulated streaming |
| fast beam search | 640ms | 3.81 | 3.34 | 3.46 | 4.58 | 3.85 | --epoch 30 --avg 12 | chunk-wise |
| greedy search | 640ms | 3.92 | 3.38 | 3.65 | 4.31 | 3.88 | --epoch 30 --avg 12 | simulated streaming |
| greedy search | 640ms | 3.98 | 3.38 | 3.64 | 4.54 | 4.01 | --epoch 30 --avg 12 | chunk-wise |
| modified beam search | 640ms | 3.72 | 3.26 | 3.39 | 4.10 | 3.65 | --epoch 30 --avg 12 | simulated streaming |
| modified beam search | 640ms | 3.78 | 3.32 | 3.45 | 4.81 | 3.81 | --epoch 30 --avg 12 | chunk-wise |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command was:
```bash
./pruned_transducer_stateless7_streaming/train.py \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --world-size 8 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2_pad30 \
  --max-duration 375 \
  --transcript-mode fluent \
  --lang data/lang_char \
  --manifest-dir /mnt/host/corpus/csj/fbank \
  --pad-feature 30 \
  --musan-dir /mnt/host/corpus/musan/musan/fbank
```

The simulated streaming decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2_pad30 \
            --epoch 30 \
            --avg 12 \
            --max-duration 350 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode fluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_fluent_2_pad30/github/sim_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --pad-feature 30 \
            --gpu 1
    done
done
```

The streaming chunk-wise decoding command was:
```bash
for chunk in 64 32; do
    for m in greedy_search fast_beam_search modified_beam_search; do
        python pruned_transducer_stateless7_streaming/streaming_decode.py \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --exp-dir pruned_transducer_stateless7_streaming/exp_fluent_2_pad30 \
            --epoch 30 \
            --avg 12 \
            --max-duration 350 \
            --decoding-method $m \
            --manifest-dir /mnt/host/corpus/csj/fbank \
            --lang data/lang_char \
            --transcript-mode fluent \
            --res-dir pruned_transducer_stateless7_streaming/exp_fluent_2_pad30/github/stream_"$chunk"_"$m" \
            --decode-chunk-len $chunk \
            --gpu 3 \
            --num-decode-streams 40
    done
done
```

#### Comparing disfluent to fluent

$$ \texttt{CER}^{f}_d = \frac{\texttt{sub}_f + \texttt{ins} + \texttt{del}_f}{N_f} $$

This comparison evaluates the disfluent model on the fluent transcript (calculated by `disfluent_recogs_to_fluent.py`), forgiving the disfluent model's mistakes on fillers and partial words. It is meant as an illustrative metric only, so that the disfluent and fluent models can be compared.

| decoding method | chunk size | eval1 (d vs f) | eval2  (d vs f) | eval3  (d vs f) | excluded  (d vs f) | valid  (d vs f) | decoding mode |
| --------------- | ---------- | -------------- | --------------- | -------------- | -------------------- | --------------- | ----------- |
| fast beam search | 320ms | 4.54 vs 4.19 | 3.44 vs 3.63 | 3.56 vs 3.77 | 4.22 vs 4.43 | 4.22 vs 4.09 | simulated streaming |
| fast beam search | 320ms | 4.48 vs 4.06 | 3.41 vs 3.55 | 3.65 vs 3.66 | 4.26 vs 4.7 | 4.08 vs 4.04 | chunk-wise |
| greedy search | 320ms | 4.53 vs 4.22 | 3.48 vs 3.62 | 3.69 vs 3.82 | 4.38 vs 4.45 | 4.05 vs 3.98 | simulated streaming |
| greedy search | 320ms | 4.53 vs 4.13 | 3.46 vs 3.61 | 3.71 vs 3.85 | 4.48 vs 4.67 | 4.12 vs 4.05 | chunk-wise |
| modified beam search | 320ms | 4.45 vs 4.02 | 3.38 vs 3.43 | 3.57 vs 3.62 | 4.19 vs 4.43 | 4.04 vs 3.81 | simulated streaming |
| modified beam search | 320ms | 4.44 vs 3.97 | 3.47 vs 3.43 | 3.56 vs 3.59 | 4.28 vs 4.99 | 4.04 vs 3.88 | chunk-wise |
| fast beam search | 640ms | 4.14 vs 3.8 | 3.12 vs 3.31 | 3.38 vs 3.55 | 3.72 vs 4.16 | 3.81 vs 3.9 | simulated streaming |
| fast beam search | 640ms | 4.05 vs 3.81 | 3.23 vs 3.34 | 3.36 vs 3.46 | 3.65 vs 4.58 | 3.78 vs 3.85 | chunk-wise |
| greedy search | 640ms | 4.1 vs 3.92 | 3.17 vs 3.38 | 3.5 vs 3.65 | 3.87 vs 4.31 | 3.77 vs 3.88 | simulated streaming |
| greedy search | 640ms | 4.41 vs 3.98 | 3.56 vs 3.38 | 3.69 vs 3.64 | 4.26 vs 4.54 | 4.16 vs 4.01 | chunk-wise |
| modified beam search | 640ms | 4 vs 3.72 | 3.08 vs 3.26 | 3.33 vs 3.39 | 3.75 vs 4.1 | 3.71 vs 3.65 | simulated streaming |
| modified beam search | 640ms | 5.05 vs 3.78 | 4.22 vs 3.32 | 4.26 vs 3.45 | 5.02 vs 4.81 | 4.73 vs 3.81 | chunk-wise |
| average (d - f) |  | 0.43 | -0.02 | -0.02 | -0.34 | 0.13 |  |
