## Results

### XBMU-AMDO31 BPE training result (Stateless Transducer)

#### Pruned transducer stateless 5

[./pruned_transducer_stateless5](./pruned_transducer_stateless5)

It uses pruned RNN-T.

A pre-trained model and decoding logs can be found at <https://huggingface.co/syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 87801200, i.e., 87.8 M

|                        | test | dev  | comment                               |
|------------------------|------|------|---------------------------------------|
| greedy search          | 11.06| 11.73| --epoch 28 --avg 23 --max-duration 600|
| beam search            | 10.64| 11.42| --epoch 28 --avg 23 --max-duration 600|
| modified beam search   | 10.57| 11.24| --epoch 28 --avg 23 --max-duration 600|


Training command is:

```bash
cd egs/xbmu_amdo31/ASR
./prepare.sh

export CUDA_VISIBLE_DEVICES="0"

./pruned_transducer_stateless5/train.py
```

**Caution**: It uses `--context-size=1`.


The decoding command is:
```bash
for method in greedy_search beam_search modified_beam_search;
do
./pruned_transducer_stateless5/decode.py \
    --epoch 28 \
    --avg 23 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 600 \
    --decoding-method $method
done
```

### pruned_transducer_stateless7 (zipformer)

See <https://github.com/k2-fsa/icefall/pull/672> for more details.

[pruned_transducer_stateless7](./pruned_transducer_stateless7)

You can find a pretrained model, training logs, decoding logs, and decoding
results at:
<https://huggingface.co/syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless7-2022-12-02>

You can use <https://github.com/k2-fsa/sherpa> to deploy it.

Number of model parameters: 70369391, i.e., 70.37 M

|                      | test | dev  | comment                                |
|----------------------|------|------|----------------------------------------|
| greedy search        | 10.06| 10.59| --epoch 23 --avg 11 --max-duration 600 |
| beam search          | 9.77 | 10.11| --epoch 23 --avg 11 --max-duration 600 |
| modified beam search | 9.7  | 10.12| --epoch 23 --avg 11 --max-duration 600 |

The training commands are:
```bash
export CUDA_VISIBLE_DEVICES="0"

./pruned_transducer_stateless7/train.py
```

The decoding commands are:
```bash
for m in greedy_search beam_search modified_beam_search; do
  for epoch in 23; do
    for avg in 11; do
      ./pruned_transducer_stateless7/decode.py \
          --epoch $epoch \
          --avg $avg \
          --exp-dir ./pruned_transducer_stateless7/exp \
          --max-duration 600 \
          --decoding-method $m
    done
  done
done
```
