# Results

## zipformer (zipformer + pruned stateless transducer)

See <https://github.com/k2-fsa/icefall/pull/1746> for more details.

[zipformer](./zipformer)

### Non-streaming

#### normal-scaled model, number of model parameters: 65549011, i.e., 65.55 M

You can find a pretrained model, training logs, decoding logs, and decoding results at:
<https://huggingface.co/zrjin/icefall-asr-libritts-zipformer-2024-10-20>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

| decoding method      | test-clean | test-other | comment            |
|----------------------|------------|------------|--------------------|
| greedy_search        | 2.83       | 5.91       | --epoch 30 --avg 5 |
| modified_beam_search | 2.80       | 5.87       | --epoch 30 --avg 5 |
| fast_beam_search     | 2.87       | 5.86       | --epoch 30 --avg 5 |
| greedy_search        | 2.76       | 5.68       | --epoch 40 --avg 16|
| modified_beam_search | 2.74       | 5.66       | --epoch 40 --avg 16|
| fast_beam_search     | 2.75       | 5.67       | --epoch 40 --avg 16|
| greedy_search        | 2.74       | 5.67       | --epoch 50 --avg 30|
| modified_beam_search | 2.73       | 5.58       | --epoch 50 --avg 30|
| fast_beam_search     | 2.78       | 5.61       | --epoch 50 --avg 30|


The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1"
./zipformer/train.py \
  --world-size 2 \
  --num-epochs 50 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 0 \
  --full-libri 1 \
  --max-duration 3600
```
This was used on 2 Nvidia A800 GPUs, you'll need to adjust the `CUDA_VISIBLE_DEVICES`, `--world-size` and `--max-duration` according to your hardware.

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search fast_beam_search; do
  ./zipformer/decode.py \
    --epoch 50 \
    --avg 30 \
    --use-averaged-model 1 \
    --exp-dir ./zipformer/exp \
    --max-duration 600 \
    --decoding-method $m
done
```
