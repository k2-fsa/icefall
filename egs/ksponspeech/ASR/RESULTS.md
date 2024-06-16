## Results

### Streaming Zipformer-Transducer (Pruned Stateless Transducer + Streaming Zipformer)

#### [pruned_transducer_stateless7_streaming](./pruned_transducer_stateless7_streaming)

Number of model parameters: 79,022,891, i.e., 79.02 M

##### Training on KsponSpeech (with MUSAN)

Model: [johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12](https://huggingface.co/johnBamma/icefall-asr-ksponspeech-pruned-transducer-stateless7-streaming-2024-06-12)

The CERs are:

| decoding method      | chunk size | eval_clean | eval_other | comment             | decoding mode        |
|----------------------|------------|------------|------------|---------------------|----------------------|
| greedy search        | 320ms      | 10.21      | 11.07      | --epoch 30 --avg 9  | simulated streaming  |
| greedy search        | 320ms      | 10.22      | 11.07      | --epoch 30 --avg 9  | chunk-wise           |
| fast beam search     | 320ms      | 10.21      | 11.04      | --epoch 30 --avg 9  | simulated streaming  |
| fast beam search     | 320ms      | 10.25      | 11.08      | --epoch 30 --avg 9  | chunk-wise           |
| modified beam search | 320ms      | 10.13      | 10.88      | --epoch 30 --avg 9  | simulated streaming  |
| modified beam search | 320ms      | 10.1       | 10.93      | --epoch 30 --avg 9  | chunk-size           |
| greedy search        | 640ms      | 9.94       | 10.82      | --epoch 30 --avg 9  | simulated streaming  |
| greedy search        | 640ms      | 10.04      | 10.85      | --epoch 30 --avg 9  | chunk-wise           |
| fast beam search     | 640ms      | 10.01      | 10.81      | --epoch 30 --avg 9  | simulated streaming  |
| fast beam search     | 640ms      | 10.04      | 10.7       | --epoch 30 --avg 9  | chunk-wise           |
| modified beam search | 640ms      | 9.91       | 10.72      | --epoch 30 --avg 9  | simulated streaming  |
| modified beam search | 640ms      | 9.92       | 10.72      | --epoch 30 --avg 9  | chunk-size           |

Note: `simulated streaming` indicates feeding full utterance during decoding using `decode.py`,
while `chunk-size` indicates feeding certain number of frames at each time using `streaming_decode.py`.

The training command is:

```bash
./pruned_transducer_stateless7_streaming/train.py \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir pruned_transducer_stateless7_streaming/exp \
    --max-duration 750 \
    --enable-musan True
```

The simulated streaming decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search fast_beam_search modified_beam_search; do
  ./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 30 \
    --avg 9 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method $m
done
```

The streaming chunk-size decoding command (e.g., chunk-size=320ms) is:
```bash
for m in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless7_streaming/streaming_decode.py \
    --epoch 30 \
    --avg 9 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --decoding-method $m \
    --decode-chunk-len 32 \
    --num-decode-streams 2000
done
```