## Results

### Aishell2 char-based training results 

#### Zipformer

[./zipformer](./zipformer)

It's reworked Zipformer with Pruned RNNT loss, note that results below are produced by model trained on data without speed perturbation applied.

**⚠️ If you prefer to have the speed perturbation disabled, please manually set `--perturb-speed` to `False` for `./local/compute_fbank_aishell.py` in the `prepare.sh` script.**

|                                       |  dev-ios  | test-ios | comment                      |
|---------------------------------------|---------|----------|----------------------------------|
|          greedy search                |  5.58   |  5.94    | --epoch 25, --avg 5, --max-duration 200 |
| modified beam search (set as default) |  5.45   |  5.86    | --epoch 25, --avg 5, --max-duration 200 |
| fast beam search (set as default)     |  5.52   |  5.91    | --epoch 25, --avg 5, --max-duration 200 |
| fast beam search oracle               |  1.65   |  1.71    | --epoch 25, --avg 5, --max-duration 200 |
| fast beam search nbest LG             |  6.14   |  6.72    | --epoch 25, --avg 5, --max-duration 200 |

The training command for reproducing is given below:

```bash
./prepare.sh # before setting --perturb-speed to False in the prepare.sh

export CUDA_VISIBLE_DEVICES="0,1"

./zipformer/train.py \
  --world-size 2 \
  --lang-dir data/lang_char \
  --num-epochs 25 \
  --start-epoch 1 \
  --max-duration 1000 \
  --use-fp16 1
```

The decoding command is:
```bash
for method in greedy_search modified_beam_search fast_beam_search fast_beam_search_nbest_oracle fast_beam_search_LG; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 25 \
    --avg 5 \
    --exp-dir ./zipformer/exp \
    --decoding-method $method \
done
```

#### Pruned transducer stateless 5

Using the codes from this commit https://github.com/k2-fsa/icefall/pull/465.

When training with context size equals to 1, the WERs are

|                                    |  dev-ios  | test-ios | comment                      |
|------------------------------------|-------|----------|----------------------------------|
|          greedy search             | 5.57  | 5.89     | --epoch 25, --avg 5, --max-duration 600  |
| modified beam search (beam size 4) | 5.32  | 5.56     | --epoch 25, --avg 5, --max-duration 600  |
| fast beam search (set as default)  | 5.5   |  5.78    | --epoch 25, --avg 5, --max-duration 600 |
| fast beam search nbest             | 5.46  |  5.74    | --epoch 25, --avg 5, --max-duration 600 |
| fast beam search oracle            | 1.92  |  2.2     | --epoch 25, --avg 5, --max-duration 600 |
| fast beam search nbest LG          | 5.59  |  5.93    | --epoch 25, --avg 5, --max-duration 600 |

The training command for reproducing is given below:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --lang-dir data/lang_char \
  --num-epochs 40 \
  --start-epoch 1 \
  --exp-dir /result \
  --max-duration 300 \
  --use-fp16 0 \
  --num-encoder-layers 24 \
  --dim-feedforward 1536 \
  --nhead 8 \
  --encoder-dim 384 \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --context-size 1
```

The decoding command is:
```bash
for method in greedy_search modified_beam_search fast_beam_search fast_beam_search_nbest  fast_beam_search_nbest_oracle fast_beam_search_nbest_LG; do
  ./pruned_transducer_stateless5/decode.py \
    --epoch 25 \
    --avg 5 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 24 \
    --dim-feedforward 1536 \
    --nhead 8 \
    --encoder-dim 384 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --context-size 1 \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5 \
    --context-size 1 \
    --use-averaged-model True
done
```
The tensorboard training log can be found at
https://tensorboard.dev/experiment/RXyX4QjQQVKjBS2eQ2Qajg/#scalars

A pre-trained model and decoding logs can be found at <https://huggingface.co/yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-B-2022-07-12>

When training with context size equals to 2, the WERs are

|                                    |  dev-ios  | test-ios | comment                      |
|------------------------------------|-------|----------|----------------------------------|
|          greedy search             | 5.47  |  5.81    | --epoch 25, --avg 5, --max-duration 600  |
| modified beam search (beam size 4) | 5.38  |  5.61    | --epoch 25, --avg 5, --max-duration 600  |
| fast beam search (set as default)  | 5.36  |  5.61    | --epoch 25, --avg 5, --max-duration 600  |
| fast beam search nbest             | 5.37  |  5.6     | --epoch 25, --avg 5, --max-duration 600 |
| fast beam search oracle            | 2.04  |  2.2     | --epoch 25, --avg 5, --max-duration 600 |
| fast beam search nbest LG          | 5.59  |  5.82     | --epoch 25, --avg 5, --max-duration 600 |

The tensorboard training log can be found at
https://tensorboard.dev/experiment/5AxJ8LHoSre8kDAuLp4L7Q/#scalars

A pre-trained model and decoding logs can be found at <https://huggingface.co/yuekai/icefall-asr-aishell2-pruned-transducer-stateless5-A-2022-07-12>
