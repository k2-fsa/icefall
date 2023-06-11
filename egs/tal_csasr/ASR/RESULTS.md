## Results

#### Pruned transducer stateless 7 (zipformer)

See <https://github.com/k2-fsa/icefall/pull/1033>

[./pruned_transducer_stateless7_bbpe](./pruned_transducer_stateless7_bbpe)

**Note**: The modeling units are byte level BPEs

The best results I have gotten are:

Vocab size | greedy (dev & test) | modified beam search (dev & test) |  |
-- | -- | -- | --
500  | 6.88 & 6.98 | 6.87 & 6.94 | --epoch 35 --avg 26

The training command:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_bbpe/train.py \
  --world-size 4 \
  --start-epoch 1 \
  --num-epochs 35 \
  --use-fp16 1 \
  --max-duration 800 \
  --bbpe-model data/lang_bbpe_500/bbpe.model \
  --exp-dir pruned_transducer_stateless7_bbpe/exp \
  --master-port 12535
```

The decoding command:

```
 ./pruned_transducer_stateless7_bbpe/decode.py \
   --epoch 35 \
   --avg 26 \
   --exp-dir ./pruned_transducer_stateless7_bbpe/exp \
   --max-sym-per-frame 1 \
   --bpe-model data/lang_bbpe_500/bbpe.model \
   --max-duration 2000 \
   --decoding-method greedy_search  # modified_beam_search
```

The pretrained model is available at:  https://huggingface.co/pkufool/icefall_asr_tal_csasr_pruned_transducer_stateless7_bbpe


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
|fast_beam_search(use-averaged-model=True) | 30 | 24 | 7.18 | 7.27|
|greedy_search | 348000 | 30 | 7.46 | 7.54|
|modified_beam_search | 348000 | 30 | 7.24 | 7.36|
|fast_beam_search | 348000 | 30 | 7.25 | 7.39 |

The results (CER(%) and WER(%)) for Chinese CER and English WER respectivly (zh: Chinese, en: English):
|decoding-method | epoch(iter) | avg | dev | dev_zh | dev_en | test | test_zh | test_en |
|--|--|--|--|--|--|--|--|--|
|greedy_search(use-averaged-model=True) | 30 | 24 | 7.30 | 6.48 | 19.19 |7.39| 6.66 | 19.13|
|modified_beam_search(use-averaged-model=True) | 30 | 24 | 7.15 | 6.35 | 18.95 | 7.22| 6.50 | 18.70 |
|fast_beam_search(use-averaged-model=True) | 30 | 24 | 7.18 | 6.39| 18.90 |  7.27| 6.55 | 18.77|

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
