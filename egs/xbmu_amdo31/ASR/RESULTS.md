## Results

### XBMU-AMDO31 BPE training result (Stateless Transducer)

#### Pruned transducer stateless 5

[./pruned_transducer_stateless5](./pruned_transducer_stateless5)

It uses pruned RNN-T.

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

A pre-trained model and decoding logs can be found at <https://huggingface.co/syzym/icefall-asr-xbmu-amdo31-pruned-transducer-stateless5-2022-11-29>
