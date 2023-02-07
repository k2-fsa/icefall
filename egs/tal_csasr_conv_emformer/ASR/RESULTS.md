## Results

### TAL_CSASR Mix Chars and BPEs training results (conv_emformer_transducer_stateless2)

#### 2022-02-07

Using the codes from this PR https://github.com/k2-fsa/icefall/pull/874.

The WERs are

|decoding-method | epoch(iter) | avg | dev | test |
|--|--|--|--|--|
|fast_beam_search(use-averaged-model=True) | 6 | 3 | 11.36 | 11.37|

The training command for reproducing is given below:

```
export  CUDA_VISIBLE_DEVICES=0,1,2,3

./conv_emformer_transducer_stateless2/train.py \
  --world-size 4 \
  --num-epochs 100 \
  --lang_dir data/lang_char \
  --start-epoch 1 \
  --start-batch 424000 \
  --use-fp16 0 \
  --master-port 12321 \
  --exp-dir exp_conv_emformer \
  --max-duration 30 \
  --num-encoder-layers 12 \
  --chunk-length 32 \
  --cnn-module-kernel 31 \
  --left-context-length 32 \
  --right-context-length 8 \
  --memory-size 32

```


The decoding command is:

```
epoch=6
avg=3
use_average_model=True

## fast_beam_search

./conv_emformer_transducer_stateless2/decode.py \
      --epoch $epoch \
      --avg $avg \
      --use-averaged-model $use_averaged \
      --exp-dir exp_conv_emformer \
      --max-duration 200 \
      --num-encoder-layers 12 \
      --chunk-length 32 \
      --cnn-module-kernel 31 \
      --left-context-length 32 \
      --right-context-length 8 \
      --memory-size 32 \
      --decoding-method fast_beam_search \
      --beam 4 \
      --max-contexts 4 \
      --max-states 8
fi

```

A pre-trained model and decoding logs can be found at <https://huggingface.co/xuancaiqisehua/icefall_asr_tal-csasr_conv_emformer_transducer_stateless2>
