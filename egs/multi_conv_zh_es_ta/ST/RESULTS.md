## Zipformer Multi-joiner ST


### For offline model training:  

You can find a pretrained model, training logs, decoding logs, and decoding results at: https://huggingface.co/AmirHussein/HENT-SRT/tree/main/zipformer_multijoiner_st


| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | modified beam search | 41.6      | 16.3       | --epoch 25, --avg 13, beam(20),  |
| hkust      | modified beam search | 23.8      | 10.4       | --epoch 25, --avg 13, beam(20),  |
| fisher\_sp | modified beam search | 18.0      | 31.0       | --epoch 25, --avg 13, beam(20),  |

The training command:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer_multijoiner_st/train.py \
  --base-lr 0.045 \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer_multijoiner_st/exp-multi-joiner-pbe4k\
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024,1024,1024,768 \
  --encoder-dim 192,256,384,512,384,384,384,256 \
  --encoder-unmasked-dim 192,192,256,256,256,256,256,192 \
  --downsampling-factor 1,2,4,8,8,4,4,2\
  --cnn-module-kernel 31,31,15,15,15,15,31,31 \
  --num-heads 4,4,4,8,8,8,4,4 \
  --bpe-st-model data/lang_st_bpe_4000/bpe.model \
  --max-duration 400 \
  --prune-range 10 \
  --warm-step 10000 \
  --lr-epochs 6 \
  --use-hat False
  ```

Decodeing command:
```
 ./zipformer_multijoiner_st/decode.py \
        --exp-dir ./zipformer_multijoiner_st/exp-multi-joiner-pbe4k \
        --epoch 25 \
        --avg 13 \
        --beam-size 20 \
        --max-duration 600 \
        --decoding-method modified_beam_search \
        --bpe-st-model data/lang_st_bpe_4000/bpe.model \
        --bpe-model data/lang_bpe_5000/bpe.model \
        --num-encoder-layers 2,2,2,2,2,2,2,2 \
        --feedforward-dim  512,768,1024,1024,1024,1024,1024,768 \
        --encoder-dim  192,256,384,512,384,384,384,256 \
        --encoder-unmasked-dim 192,192,256,256,256,256,256,192 \
        --downsampling-factor 1,2,4,8,8,4,4,2 \
        --cnn-module-kernel 31,31,15,15,15,15,31,31 \
        --num-heads 4,4,4,8,8,8,4,4 \
        --use-averaged-model True
```

### For streaming  model training:  

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | greedy search |   44.1    |    6.0    | --epoch 25, --avg 13  |
| hkust      | greedy search |    27.4  |    3.7   | --epoch 25, --avg 13  |
| fisher\_sp | greedy search |  19.9     |  16.3      | --epoch 25, --avg 13  |

The training command:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./zipformer_multijoiner_st/train.py \
  --base-lr 0.045 \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer_multijoiner_st/exp-multi-joiner-pbe4k_causal\
  --causal 1 \
  --num-encoder-layers 2,2,2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024,1024,1024,768 \
  --encoder-dim 192,256,384,512,384,384,384,256 \
  --encoder-unmasked-dim 192,192,256,256,256,256,256,192 \
  --downsampling-factor 1,2,4,8,8,4,4,2\
  --cnn-module-kernel 31,31,15,15,15,15,31,31 \
  --num-heads 4,4,4,8,8,8,4,4 \
  --bpe-st-model data/lang_st_bpe_4000/bpe.model \
  --max-duration 400 \
  --prune-range 10 \
  --warm-step 10000 \
  --lr-epochs 6 \
  --use-hat False
  ```

Decodeing command:
```
 ./zipformer_multijoiner_st/decode.py \
        --exp-dir ./zipformer_multijoiner_st/exp-multi-joiner-pbe4k \
        --causal 1  \
        --epoch 25 \
        --avg 13 \
        --beam-size 20 \
        --max-duration 600 \
        --decoding-method modified_beam_search \
        --bpe-st-model data/lang_st_bpe_4000/bpe.model \
        --bpe-model data/lang_bpe_5000/bpe.model \
        --num-encoder-layers 2,2,2,2,2,2,2,2 \
        --feedforward-dim  512,768,1024,1024,1024,1024,1024,768 \
        --encoder-dim  192,256,384,512,384,384,384,256 \
        --encoder-unmasked-dim 192,192,256,256,256,256,256,192 \
        --downsampling-factor 1,2,4,8,8,4,4,2 \
        --cnn-module-kernel 31,31,15,15,15,15,31,31 \
        --num-heads 4,4,4,8,8,8,4,4 \
        --use-averaged-model True \
        --decoding-method greedy_search \
        --chunk-size 64 \
        --left-context-frames 128 \
        --use-hat False \
        --max-sym-per-frame 20 
```


## Hent-SRT offline

You can find a pretrained model, training logs, decoding logs, and decoding results at: https://huggingface.co/AmirHussein/HENT-SRT/tree/main/hent_srt

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | modified beam search | 41.4      | 20.6       | --epoch 20, --avg 13, beam(20),  BP 1 |
| hkust      | modified beam search | 22.8      | 14.7       | --epoch 20, --avg 13, beam(20),  BP 1 |
| fisher\_sp | modified beam search | 17.8      | 33.7       | --epoch 20, --avg 13, beam(20),  BP 1 |


### First pretrain the offline CR-CTC ASR

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./hent_srt/train.py \
    --base-lr 0.055 \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir hent_srt/exp-asr\
    --causal 0 \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1024,1024 \
    --encoder-dim 192,256,384,512,384 \
    --encoder-unmasked-dim 192,192,256,256,256 \
    --downsampling-factor 1,2,4,8,4 \
    --cnn-module-kernel 31,31,15,15,15 \
    --num-heads 4,4,4,8,8 \
    --st-num-encoder-layers 2,2,2,2,2 \
    --st-feedforward-dim 512,512,256,256,256 \
    --st-encoder-dim 512,384,256,256,256 \
    --st-encoder-unmasked-dim 256,256,256,256,192 \
    --st-downsampling-factor 4,4,4,4,4 \
    --st-cnn-module-kernel 15,31,31,15,15 \
    --st-num-heads 4,4,8,8,8 \
    --bpe-st-model data/lang_st_bpe_4000/bpe.model \
    --bpe-model data/lang_bpe_5000/bpe.model \
    --manifest-dir data/fbank \
    --max-duration 350 \
    --prune-range 10 \
    --warm-step 8000 \
    --ctc-loss-scale 0.2 \
    --enable-spec-aug 0 \
    --cr-loss-scale 0.2 \
    --time-mask-ratio 2.5 \
    --use-asr-cr-ctc 1 \
    --use-ctc 1 \
    --lr-epochs 6 \
    --use-hat False \
    --use-st-joiner False 
```

### Train ST with a Pretrained ASR Initialization

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./hent_srt/train.py \
  --base-lr 0.045 \
  --world-size 4 \
  --num-epochs 25 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir hent_srt/exp-st \
  --model-init-ckpt hent_srt/exp-asr/best-valid-loss.pt \
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024 \
  --encoder-dim 192,256,384,512,384 \
  --encoder-unmasked-dim 192,192,256,256,256 \
  --downsampling-factor 1,2,4,8,4 \
  --cnn-module-kernel 31,31,15,15,15 \
  --num-heads 4,4,4,8,8 \
  --st-num-encoder-layers 2,2,2,2,2 \
  --st-feedforward-dim 512,512,256,256,256 \
  --st-encoder-dim 384,512,256,256,256 \
  --st-encoder-unmasked-dim 256,256,256,256,192 \
  --st-downsampling-factor 1,2,4,4,4  \
  --st-cnn-module-kernel 15,31,31,15,15 \
  --st-num-heads 8,8,8,8,8 \
  --output-downsampling-factor 2 \
  --st-output-downsampling-factor 1 \
  --bpe-st-model data/lang_st_bpe_4000/bpe.model \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --manifest-dir data/fbank \
  --max-duration 200 \
  --prune-range 5 \
  --st-prune-range 10 \
  --warm-step 10000 \
  --ctc-loss-scale 0.1 \
  --st-ctc-loss-scale 0.1 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.05 \
  --st-cr-loss-scale 0.05 \
  --time-mask-ratio 2.5 \
  --use-asr-cr-ctc 1 \
  --use-ctc 1 \
  --use-st-cr-ctc 1 \
  --use-st-ctc 1 \
  --lr-epochs 6 \
  --use-hat False \
  --use-st-joiner True
```

### Decode offline Hent-SRT

```
./hent_srt/decode.py \
    --epoch 20 --avg 13 --use-averaged-model True \
    --beam-size 20 \
    --causal 0 \
    --exp-dir hent_srt/exp-st \
    --bpe-model data/lang_bpe_5000/bpe.model \
    --bpe-st-model data/lang_st_bpe_4000/bpe.model \
    --output-downsampling-factor 2 \
    --st-output-downsampling-factor 1 \
    --max-duration 800 \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1024,1024 \
    --encoder-dim 192,256,384,512,384 \
    --encoder-unmasked-dim 192,192,256,256,256 \
    --downsampling-factor 1,2,4,8,4 \
    --cnn-module-kernel 31,31,15,15,15 \
    --num-heads 4,4,4,8,8 \
    --st-num-encoder-layers 2,2,2,2,2 \
    --st-feedforward-dim 512,512,256,256,256 \
    --st-encoder-dim 384,512,256,256,256 \
    --st-encoder-unmasked-dim 256,256,256,256,192 \
    --st-downsampling-factor 1,2,4,4,4  \
    --st-cnn-module-kernel 15,31,31,15,15 \
    --st-num-heads 8,8,8,8,8 \
    --decoding-method modified_beam_search \
    --use-st-joiner True \
    --use-hat-decode False \
    --use-ctc 1 \
    --use-st-ctc 1 \
    --st-blank-penalty 1
```


## Hent-SRT streaming

| Dataset    | Decoding method      | test WER | test BLEU | comment                                         |
| ---------- | -------------------- | -------- | --------- | ----------------------------------------------- |
| iwslt\_ta  | greedy search | 46.2      | 17.3       | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20  |
| hkust      | greedy search | 27.3      | 11.2       | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20|
| fisher\_sp | greedy search | 22.7      | 30.8     | --epoch 20, --avg 13, BP 2, chunk-size 64, left-context-frames 128, max-sym-per-frame 20 |

### First pretrain the streaming CR-CTC ASR

# CR-CTC ASR streaming
```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./hent_srt/train.py \
    --base-lr 0.055 \
    --world-size 4 \
    --num-epochs 30 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir hent_srt/exp-asr_causal\
    --causal 1 \
    --num-encoder-layers 2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1024,1024 \
    --encoder-dim 192,256,384,512,384 \
    --encoder-unmasked-dim 192,192,256,256,256 \
    --downsampling-factor 1,2,4,8,4 \
    --cnn-module-kernel 31,31,15,15,15 \
    --num-heads 4,4,4,8,8 \
    --st-num-encoder-layers 2,2,2,2,2 \
    --st-feedforward-dim 512,512,256,256,256 \
    --st-encoder-dim 512,384,256,256,256 \
    --st-encoder-unmasked-dim 256,256,256,256,192 \
    --st-downsampling-factor 4,4,4,4,4 \
    --st-cnn-module-kernel 15,31,31,15,15 \
    --st-num-heads 4,4,8,8,8 \
    --bpe-st-model data/lang_st_bpe_4000/bpe.model \
    --bpe-model data/lang_bpe_5000/bpe.model \
    --manifest-dir data/fbank \
    --max-duration 250 \
    --prune-range 10 \
    --warm-step 8000 \
    --ctc-loss-scale 0.2 \
    --enable-spec-aug 0 \
    --cr-loss-scale 0.2 \
    --time-mask-ratio 2.5 \
    --use-asr-cr-ctc 1 \
    --use-ctc 1 \
    --lr-epochs 6 \
    --use-hat False \
    --use-st-joiner False 
```


### Train streaming ST with a Pretrained ASR Initialization
```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
./hent_srt/train.py \
  --base-lr 0.045 \
  --world-size 4 \
  --num-epochs 25 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir hent_srt/exp-st_causal \
  --model-init-ckpt hent_srt/exp-asr_causal/best-valid-loss.pt \
  --causal 1 \
  --num-encoder-layers 2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1024,1024 \
  --encoder-dim 192,256,384,512,384 \
  --encoder-unmasked-dim 192,192,256,256,256 \
  --downsampling-factor 1,2,4,8,4 \
  --cnn-module-kernel 31,31,15,15,15 \
  --num-heads 4,4,4,8,8 \
  --st-num-encoder-layers 2,2,2,2,2 \
  --st-feedforward-dim 512,512,256,256,256 \
  --st-encoder-dim 384,512,256,256,256 \
  --st-encoder-unmasked-dim 256,256,256,256,192 \
  --st-downsampling-factor 1,2,4,4,4  \
  --st-cnn-module-kernel 15,31,31,15,15 \
  --st-num-heads 8,8,8,8,8 \
  --output-downsampling-factor 2 \
  --st-output-downsampling-factor 1 \
  --bpe-st-model data/lang_st_bpe_4000/bpe.model \
  --bpe-model data/lang_bpe_5000/bpe.model \
  --manifest-dir data/fbank \
  --max-duration 200 \
  --prune-range 5 \
  --st-prune-range 10 \
  --warm-step 10000 \
  --ctc-loss-scale 0.1 \
  --st-ctc-loss-scale 0.1 \
  --enable-spec-aug 0 \
  --cr-loss-scale 0.05 \
  --st-cr-loss-scale 0.05 \
  --time-mask-ratio 2.5 \
  --use-asr-cr-ctc 1 \
  --use-ctc 1 \
  --use-st-cr-ctc 1 \
  --use-st-ctc 1 \
  --lr-epochs 6 \
  --use-hat False \
  --use-st-joiner True 
  ```

  ### Decode streaming Hent-SRT
```
./hent_srt/decode.py \
        --epoch 20 --avg 13 --use-averaged-model True \
        --causal 1 \
        --exp-dir hent_srt/exp-st_causal \
        --bpe-model data/lang_bpe_5000/bpe.model \
        --bpe-st-model data/lang_st_bpe_4000/bpe.model \
        --output-downsampling-factor 2 \
        --st-output-downsampling-factor 1 \
        --max-duration 800 \
        --num-encoder-layers 2,2,2,2,2 \
        --feedforward-dim 512,768,1024,1024,1024 \
        --encoder-dim 192,256,384,512,384 \
        --encoder-unmasked-dim 192,192,256,256,256 \
        --downsampling-factor 1,2,4,8,4 \
        --cnn-module-kernel 31,31,15,15,15 \
        --num-heads 4,4,4,8,8 \
        --st-num-encoder-layers 2,2,2,2,2 \
        --st-feedforward-dim 512,512,256,256,256 \
        --st-encoder-dim 384,512,256,256,256 \
        --st-encoder-unmasked-dim 256,256,256,256,192 \
        --st-downsampling-factor 1,2,4,4,4  \
        --st-cnn-module-kernel 15,31,31,15,15 \
        --st-num-heads 8,8,8,8,8 \
        --decoding-method greedy_search \
        --use-st-joiner True \
        --use-hat-decode False \
        --use-ctc 1 \
        --use-st-ctc 1 \
        --st-blank-penalty 2 \
        --chunk-size 64 \
        --left-context-frames 128 \
        --use-hat False --max-sym-per-frame 20 
```