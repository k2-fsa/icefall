# Results



### IWSLT Tunisian training results (Stateless Pruned Transducer)

#### 2023-06-01


|    Decoding method                 |     dev WER     |    test WER    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 47.6      | 51.2       | --epoch 20, --avg 13  |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"


  
./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless5/exp \
  --max-duration 300 \
  --num-buckets 50
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/yBijWJSPSGuBqMwTZ509lA/

The decoding command is:
```
for method in modified_beam_search; do
    ./pruned_transducer_stateless5/decode.py \
    --epoch 15 \
    --beam-size 20 \
    --avg 5 \
    --exp-dir ./pruned_transducer_stateless5/exp \
    --max-duration 400 \
    --decoding-method modified_beam_search \
    --max-sym-per-frame 1 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 8 \
    --encoder-dim 256 \
    --decoder-dim 256 \
    --joiner-dim 256 \
    --use-averaged-model true
done
```

### IWSLT Tunisian training results  (Zipformer)

#### 2023-06-01

You can find a pretrained model, training logs, decoding logs, and decoding results at: 
<https://huggingface.co/AmirHussein/zipformer-iwslt22-Ta>



|    Decoding method                 |     dev WER     |    test WER    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 40.8      | 44.1       | --epoch 20, --avg 13  |

To reproduce the above result, use the following commands for training:

# Note: the model was trained on V-100 32GB GPU

```
export CUDA_VISIBLE_DEVICES="0,1"
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --causal 0 \
  --num-encoder-layers 2,2,2,2,2,2 \
  --feedforward-dim 512,768,1024,1536,1024,768 \
  --encoder-dim 192,256,384,512,384,256 \
  --encoder-unmasked-dim 192,192,256,256,256,192 \
  --max-duration 800 \
  --prune-range 10

```

The decoding command is:

```
for method in modified_beam_search; do
  ./zipformer/decode.py \
  --epoch 20 \
  --beam-size 20 \
  --avg 13 \
  --exp-dir ./zipformer/exp\
  --max-duration 800 \
  --decoding-method $method \
 	--num-encoder-layers 2,2,2,2,2,2 \
 	--feedforward-dim 512,768,1024,1536,1024,768 \
 	--encoder-dim 192,256,384,512,384,256 \
 	--encoder-unmasked-dim 192,192,256,256,256,192 \
  --use-averaged-model true
 done
```




