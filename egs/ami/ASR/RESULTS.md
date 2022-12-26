## Results

### AMI training results (Pruned Transducer)

#### 2022-11-20

#### Zipformer (pruned_transducer_stateless7)

Zipformer encoder + non-current decoder. The decoder
contains only an embedding layer, a Conv1d (with kernel size 2) and a linear
layer (to transform tensor dim).

All the results below are using a single model that is trained by combining the following
data: IHM, IHM+reverb, SDM, and GSS-enhanced MDM. Speed perturbation and MUSAN noise
augmentation are applied on top of the pooled data.

**WERs for IHM:**

|                           | dev | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  19.25  |  17.83  | --epoch 14 --avg 8 --max-duration 500 |
| modified beam search      |  18.92  |  17.40  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  19.44  |  18.04  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

**WERs for SDM:**

|                           | dev | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  31.32  |  32.38  | --epoch 14 --avg 8 --max-duration 500 |
| modified beam search      |  31.25  |  32.21  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  31.11  |  32.10  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

**WERs for GSS-enhanced MDM:**

|                           | dev | test | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             |  22.05  |  22.93  | --epoch 14 --avg 8 --max-duration 500 |
| modified beam search      |  21.67  |  22.43  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 |
| fast beam search          |  22.21  |  22.83  | --epoch 14 --avg 8 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7/train.py \
  --world-size 4 \
  --num-epochs 15 \
  --exp-dir pruned_transducer_stateless7/exp \
  --max-duration 150 \
  --max-cuts 150 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25 \
  --use-fp16 True
```

The decoding command is:
```
# greedy search
./pruned_transducer_stateless7/decode.py \
        --epoch 14 \
        --avg 8 \
        --exp-dir ./pruned_transducer_stateless7/exp \
        --max-duration 500 \
        --decoding-method greedy_search

# modified beam search
./pruned_transducer_stateless7/decode.py \
        --iter 105000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless7/exp \
        --max-duration 500 \
        --decoding-method modified_beam_search \
        --beam-size 4

# fast beam search
./pruned_transducer_stateless7/decode.py \
        --iter 105000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless5/exp \
        --max-duration 500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
```

Pretrained model is available at <https://huggingface.co/desh2608/icefall-asr-ami-pruned-transducer-stateless7>

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/VH10QOTBTbuYpWx994Onrg/#scalars>
