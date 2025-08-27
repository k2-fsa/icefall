# Results for train-clean-100

This page shows the WERs for test-clean/test-other using only
train-clean-100 subset as training data.

## Distillation with hubert
### 2022-05-27
Related models/log/tensorboard:
https://huggingface.co/GuoLiyong/stateless6_baseline_vs_disstillation

Following results are obtained by ./distillation_with_hubert.sh

The only differences is in pruned_transducer_stateless6/train.py.

For baseline: set enable_distillation=False

For distillation: set enable_distillation=True (the default)

Decoding method is modified beam search.
|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| baseline no vq distillation         | 7.09       | 18.88      | --epoch 20, --avg 10, --max-duration 200 |
| baseline no vq distillation         | 6.83       | 18.19      | --epoch 30, --avg 10, --max-duration 200 |
| baseline no vq distillation         | 6.73       | 17.79      | --epoch 40, --avg 10, --max-duration 200 |
| baseline no vq distillation         | 6.75       | 17.68      | --epoch 50, --avg 10, --max-duration 200 |
| distillation with hubert            | 5.82       | 15.98      | --epoch 20, --avg 10, --max-duration 200 |
| distillation with hubert            | 5.52       | 15.15      | --epoch 30, --avg 10, --max-duration 200 |
| distillation with hubert            | 5.45       | 14.94      | --epoch 40, --avg 10, --max-duration 200 |
| distillation with hubert            | 5.50       | 14.77      | --epoch 50, --avg 10, --max-duration 200 |

## Conformer encoder + embedding decoder

### 2022-02-21

Using commit `2332ba312d7ce72f08c7bac1e3312f7e3dd722dc`.

|                                     | test-clean | test-other | comment                                  |
|-------------------------------------|------------|------------|------------------------------------------|
| greedy search (max sym per frame 1) | 6.34       | 16.7       | --epoch 57, --avg 17, --max-duration 100 |
| greedy search (max sym per frame 2) | 6.34       | 16.7       | --epoch 57, --avg 17, --max-duration 100 |
| greedy search (max sym per frame 3) | 6.34       | 16.7       | --epoch 57, --avg 17, --max-duration 100 |
| modified beam search (beam size 4)  | 6.31       | 16.3       | --epoch 57, --avg 17, --max-duration 100 |


The training command for reproducing is given below:

```bash
cd egs/librispeech/ASR/
./prepare.sh
./prepare_giga_speech.sh

export CUDA_VISIBLE_DEVICES="0,1"

./transducer_stateless_multi_datasets/train.py \
  --world-size 2 \
  --num-epochs 60 \
  --start-epoch 0 \
  --exp-dir transducer_stateless_multi_datasets/exp-100-2 \
  --full-libri 0 \
  --max-duration 300 \
  --lr-factor 1 \
  --bpe-model data/lang_bpe_500/bpe.model \
  --modified-transducer-prob 0.25
  --giga-prob 0.2
```

The decoding command is given below:

```bash
for epoch in 57; do
  for avg in 17; do
    for sym in 1 2 3; do
    ./transducer_stateless_multi_datasets/decode.py \
      --epoch $epoch \
      --avg $avg \
      --exp-dir transducer_stateless_multi_datasets/exp-100-2 \
      --bpe-model ./data/lang_bpe_500/bpe.model \
      --max-duration 100 \
      --context-size 2 \
      --max-sym-per-frame $sym
    done
  done
done

epoch=57
avg=17
./transducer_stateless_multi_datasets/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir transducer_stateless_multi_datasets/exp-100-2 \
  --bpe-model ./data/lang_bpe_500/bpe.model \
  --max-duration 100 \
  --context-size 2 \
  --decoding-method modified_beam_search \
  --beam-size 4
```

The tensorboard log is available at
<https://tensorboard.dev/experiment/qUEKzMnrTZmOz1EXPda9RA/>

A pre-trained model and decoding logs can be found at
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-100h-transducer-stateless-multi-datasets-bpe-500-2022-02-21>
