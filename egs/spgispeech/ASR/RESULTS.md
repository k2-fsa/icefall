## Results

### SPGISpeech BPE training results (Pruned Transducer)

#### 2022-05-11

#### Conformer encoder + embedding decoder

Conformer encoder + non-current decoder. The decoder
contains only an embedding layer, a Conv1d (with kernel size 2) and a linear
layer (to transform tensor dim).

The WERs are

|                           | dev | val | comment                                  |
|---------------------------|------------|------------|------------------------------------------|
| greedy search             | 2.46       | 2.40       | --avg-last-n 10 --max-duration 500 |
| beam search               | 2.27       | 2.24       | --avg-last-n 10 --max-duration 500 --beam-size 4 |
| modified beam search      | 2.34       | 2.30       | --avg-last-n 10 --max-duration 500 --beam-size 4 |
| fast beam search          | 2.38       | 2.35       | --avg-last-n 10 --max-duration 500 --beam-size 4 --max-contexts 4 --max-states 8 |

**NOTE:** SPGISpeech transcripts can be prepared in `ortho` or `norm` ways, which refer to whether the
transcripts are orthographic or normalized. These WERs correspond to the normalized transcription
scenario.

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless2/train.py \
  --world-size 8 \
  --num-epochs 20 \
  --start-epoch 0 \
  --exp-dir pruned_transducer_stateless2/exp \
  --max-duration 200 \
  --prune-range 5 \
  --lr-factor 5 \
  --lm-scale 0.25 \
  --use-fp16 True
```

The tensorboard training log can be found at
<https://tensorboard.dev/experiment/ExSoBmrPRx6liMTGLu0Tgw/#scalars>

The decoding command is:
```
## fast beam search
./pruned_transducer_stateless/decode.py \
  --avg-last-n 10 \
  --exp-dir pruned_transducer_stateless/exp \
  --max-duration 500 \
  --beam-size 4 \
  --max-contexts 4 \
  --max-states 8
```
