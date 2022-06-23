## Results

### TAL_CSASR Mix Chars and BPEs training results (Pruned Transducer Stateless5)

#### 2022-06-22

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/428.

The WERs are

|decoding-method | epoch(iter) | avg | dev | test |
|--|--|--|--|--|
|greedy_search | 30 | 24 | 7.49 | 7.58|
|modified_beam_search | 30 | 24 | 7.33 | 7.38|
|fast_beam_search | 30 | 24 | 7.32 | 7.42|
|greedy_search(use-averaged-model=True) | 30 | 24 | 7.30 | 7.39|
|modified_beam_search(use-averaged-model=True) | 30 | 24 | 7.15 | 7.22|
|fast_beam_search(use-averaged-model=True) | 30 | 24 | 7.18 | 7.26|
|greedy_search | 348000 | 30 | 7.46 | 7.54|
|modified_beam_search | 348000 | 30 | 7.24 | 7.36|
|fast_beam_search | 348000 | 30 | 7.25 | 7.39 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

./pruned_transducer_stateless5/train.py \
  --world-size 6 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir data/lang_char \
  --max-duration 90
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/KaACzXOVR0OM6cy0qbN5hw/#scalars

The decoding command is:
```
epoch=30
avg=24
use_average_model=True

## greedy search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800 \
  --use-averaged-model $use_average_model

## modified beam search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 800 \
  --decoding-method modified_beam_search \
  --beam-size 4 \
  --use-averaged-model $use_average_model

## fast beam search
./pruned_transducer_stateless5/decode.py \
  --epoch $epoch \
  --avg $avg \
  --exp-dir ./pruned_transducer_stateless5/exp \
  --lang-dir ./data/lang_char \
  --max-duration 1500 \
  --decoding-method fast_beam_search \
  --beam 4 \
  --max-contexts 4 \
  --max-states 8 \
  --use-averaged-model $use_average_model
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/luomingshuang/icefall_asr_tal-csasr_pruned_transducer_stateless5>
