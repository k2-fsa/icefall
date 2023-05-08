## Results (CER)

#### 2022-12-09

#### Zipformer (pruned_transducer_stateless7)

Zipformer encoder + non-current decoder. The decoder
contains only an embedding layer, a Conv1d (with kernel size 2) and a linear
layer (to transform tensor dim).

All the results below are using a single model that is trained by combining the following
data: IHM, IHM+reverb, SDM, and GSS-enhanced MDM. Speed perturbation and MUSAN noise
augmentation are applied on top of the pooled data.

**WERs for IHM:**

|                           | eval | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  10.13  |  12.21  | --epoch 15 --avg 8 --max-duration 500 |
| modified beam search      |  9.58  |  11.53  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  9.92  |  12.07  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

**WERs for SDM:**

|                           | eval | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  23.70  |  26.41  | --epoch 15 --avg 8 --max-duration 500 |
| modified beam search      |  23.37  |  25.85  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  23.60  |  26.38  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

**WERs for GSS-enhanced MDM:**

|                           | eval | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  12.24  |  14.99  | --epoch 15 --avg 8 --max-duration 500 |
| modified beam search      |  11.82  |  14.22  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  12.30  |  14.98  | --epoch 15 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 15 \
  --exp-dir pruned_transducer_stateless7/exp \
  --max-duration 300 \
  --max-cuts 100 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25 \
  --use-fp16 True
```

The decoding command is:
```
# greedy search
./pruned_transducer_stateless7/decode.py \
        --epoch 15 \
        --avg 8 \
        --exp-dir ./pruned_transducer_stateless7/exp \
        --max-duration 500 \
        --decoding-method greedy_search

# modified beam search
./pruned_transducer_stateless7/decode.py \
        --epoch 15 \
        --avg 8 \
        --exp-dir ./pruned_transducer_stateless7/exp \
        --max-duration 500 \
        --decoding-method modified_beam_search \
        --beam-size 4

# fast beam search
./pruned_transducer_stateless7/decode.py \
        --epoch 15 \
        --avg 8 \
        --exp-dir ./pruned_transducer_stateless5/exp \
        --max-duration 500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
```

Pretrained model is available at <https://huggingface.co/desh2608/icefall-asr-alimeeting-pruned-transducer-stateless7>

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/EzmVahMMTb2YfKWXwQ2dyQ/#scalars>
