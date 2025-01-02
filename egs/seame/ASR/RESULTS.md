## Results

#### Zipformer

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 21.87      | 29.04      | --epoch 25, --avg 5, --max-duration 500 |

The training command:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./zipformer/train.py \
  --world-size 4 \
  --num-epochs 25 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp-asr-seame \
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --prune-range 10 \
  --max-duration 500
```

The decoding command:

```
 ./zipformer/decode.py \
    --epoch 25 \
    --avg 5 \
    --beam-size 10
    --exp-dir ./zipformer/exp-asr-seame \
    --max-duration 800 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1024,1024,768 \
    --encoder-dim 192,256,256,256,256,256 \
    --encoder-unmasked-dim 192,192,192,192,192,192 \
    --decoding-method modified_beam_search
```

The pretrained model is available at:  <https://huggingface.co/AmirHussein/zipformer-seame>


### Zipformer-HAT

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 22.00      | 29.92      | --epoch 20, --avg 5, --max-duration 500 |


The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

./zipformer_hat/train.py \
  --world-size 4 \
  --num-epochs 25 \
  --start-epoch 1 \
  --base-lr 0.045 \
  --lr-epochs 6 \
  --use-fp16 1 \
  --exp-dir zipformer_hat/exp \
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --prune-range 10 \
  --max-duration 500 
```

The decoding command is:
```
## modified beam search
./zipformer_hat/decode.py \
      --epoch 25 --avg 5 --use-averaged-model True \
      --beam-size 10 \
      --causal 0 \
      --exp-dir zipformer_hat/exp \
      --bpe-model data_seame/lang_bpe_4000/bpe.model \
      --max-duration 1000 \
      --num-encoder-layers 2,2,2,2,2,2 \
      --feedforward-dim 512,768,1024,1024,1024,768 \
      --encoder-dim 192,256,256,256,256,256 \
      --encoder-unmasked-dim 192,192,192,192,192,192 \
      --decoding-method modified_beam_search 
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/AmirHussein/zipformer-hat-seame>

### Zipformer-HAT-LID

|                                    |     dev    |    test    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 20.04      | 26.91      | --epoch 15, --avg 5, --max-duration 500 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

./zipformer_hat_lid/train.py \
  --world-size 4 \
  --lid True \
  --num-epochs 25 \
  --start-epoch 1 \
  --base-lr 0.045 \
  --use-fp16 1 \
  --lid-loss-scale 0.3 \
  --exp-dir zipformer_hat_lid/exp \
  --causal 0 \
  --lid-output-layer 3 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024,768 \
  --encoder-dim 192,256,256,256,256,256 \
  --encoder-unmasked-dim 192,192,192,192,192,192 \
  --lids "<en>,<zh>" \
  --prune-range 10 \
  --freeze-main-model False \
  --use-lid-encoder True \
  --use-lid-joiner True \
  --lid-num-encoder-layers 2,2,2 \
  --lid-downsampling-factor 2,4,2 \
  --lid-feedforward-dim 256,256,256 \
  --lid-num-heads 4,4,4 \
  --lid-encoder-dim 256,256,256 \
  --lid-encoder-unmasked-dim 128,128,128 \
  --lid-cnn-module-kernel 31,15,31 \
  --max-duration 500

```

The decoding command is:
```
## modified beam search
python zipformer_hat_lid/decode.py \
      --epoch $epoch --avg 5 --use-averaged-model True \
      --beam-size 10 \
      --lid False \
      --lids "<en>,<zh>" \
      --exp-dir zipformer_hat_lid/exp \
      --bpe-model data_seame/lang_bpe_4000/bpe.model \
      --max-duration 800 \
      --num-encoder-layers 2,2,2,2,2,2 \
      --feedforward-dim 512,768,1024,1024,1024,768 \
      --encoder-dim 192,256,256,256,256,256 \
      --encoder-unmasked-dim 192,192,192,192,192,192 \
      --decoding-method modified_beam_search \
      --lid-output-layer 3 \
      --use-lid-encoder True \
      --use-lid-joiner True \
      --lid-num-encoder-layers 2,2,2 \
      --lid-downsampling-factor 2,4,2 \
      --lid-feedforward-dim 256,256,256 \
      --lid-num-heads 4,4,4 \
      --lid-encoder-dim 256,256,256 \
      --lid-encoder-unmasked-dim 128,128,128 \
      --lid-cnn-module-kernel 31,15,31 
```

A pre-trained model and decoding logs can be found at <https://huggingface.co/AmirHussein/zipformer-hat-lid-seame>


