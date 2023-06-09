# data2vec-transducer

|  | test-clean | test-other |
| --- | --- | --- |
| greedy decoding | 2.88 | 6.69 |
| modified beam search | 2.76 | 6.37 |
| fast beam search | 2.82 | 6.59 |
- train command

```bash
./pruned_transducer_stateless_d2v_v2/train.py \
        --wandb False \
        --input-strategy AudioSamples \
        --enable-spec-aug False \
        --multi-optim True \
        --start-epoch 1 \ 
        --world-size 4 \ 
        --num-epochs 30 \
        --full-libri 1 \ 
        --exp-dir ./pruned_transducer_stateless_d2v_v2/d2v-T \
        --max-duration 150 \
        --freeze-finetune-updates 3000 \
        --encoder-dim 768 \
        --decoder-dim 768 \
        --joiner-dim 768 \
        --use-fp16 1 \ 
        --peak-dec-lr 0.04175 \
        --peak-enc-lr 0.0003859 \
        --accum-grads 4 \ 
        --encoder-type d2v \
        --additional-block True \
        --prune-range 10 \
        --context-size 2 \ 
        --ctc-loss-scale 0.2
```

- decode command

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless_d2v_v2/decode.py \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --additional-block True \
    --model-name epoch-27.pt \
    --exp-dir ./pruned_transducer_stateless_d2v_v2/960h_sweep_v3_388 \
    --max-duration 400 \
    --decoding-method $method \
    --max-sym-per-frame 1 \ 
    --encoder-type d2v \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768
```