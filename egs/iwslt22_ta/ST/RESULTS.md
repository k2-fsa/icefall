# Results


### IWSLT Tunisian training results (Stateless Pruned Transducer)

#### 2023-06-01


|    Decoding method                 |     dev Bleu     |    test Bleu    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 11.1	    | 9.2    | --epoch 20, --avg 10, beam(10), pruned range 5 |

The training command for reproducing is given below:

```
export CUDA_VISIBLE_DEVICES="0,1,2,3"


  
./pruned_transducer_stateless5/train.py \
  --world-size 4 \
  --num-epochs 20 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless5/exp \
  --max-duration 300 \
  --bucketing-sampler 1\
  --num-buckets 50
```

The tensorboard training log can be found at
https://tensorboard.dev/experiment/YnzQNCVDSxCvP1onrCzg9A/

The decoding command is:
```
for method in modified_beam_search; do
  for epoch in 15 20; do
    ./pruned_transducer_stateless5/decode.py \
      --epoch $epoch \
      --beam-size 20 \
      --avg 10 \
      --exp-dir ./pruned_transducer_stateless5/exp_st \
      --max-duration 300 \
      --decoding-method $method \
      --max-sym-per-frame 1 \
      --num-encoder-layers 12 \
      --dim-feedforward 1024 \
      --nhead 8 \
      --encoder-dim 256 \
      --decoder-dim 256 \
      --joiner-dim 256 \
      --use-averaged-model true
done
done
```

### IWSLT Tunisian training results  (Zipformer)

#### 2023-06-01

You can find a pretrained model, training logs, decoding logs, and decoding results at: 
<https://huggingface.co/AmirHussein/zipformer-iwslt22-Ta>



|    Decoding method                 |     dev Bleu     |    test Bleu    | comment                                  |
|------------------------------------|------------|------------|------------------------------------------|
| modified beam search               | 14.7	    | 12.4       | --epoch 20, --avg 10, beam(10),pruned range 5 |
| modified beam search               | 15.5	    | 13      | --epoch 20, --avg 10, beam(20),pruned range 5 |
| modified beam search               | 18.2	   | 14.8        | --epoch 20, --avg 10, beam(20), pruned range 10 |



To reproduce the above result, use the following commands for training:

# Note: the model was trained on V-100 32GB GPU
# ST medium model 42.5M prune-range 10
```

  ./zipformer/train.py \
    --world-size 4 \
    --num-epochs 25 \
    --start-epoch 1 \
    --use-fp16 1 \
    --exp-dir zipformer/exp-st-medium \
    --causal 0 \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --max-duration 800 \
    --prune-range 10 \
    --warm-step 5000 \
    --lr-epochs 8 \
    --base-lr 0.055 \
    --use-hat False
  
```


The decoding command is:

```
for method in modified_beam_search; do
  for epoch in 15 20; do
    ./zipformer/decode.py \
    --epoch $epoch \
    --beam-size 20 \
    --avg 10 \
    --exp-dir ./zipformer/exp-st-medium-prun10 \
    --max-duration 800 \
    --decoding-method $method \
    --num-encoder-layers 2,2,2,2,2,2 \
    --feedforward-dim 512,768,1024,1536,1024,768 \
    --encoder-dim 192,256,384,512,384,256 \
    --encoder-unmasked-dim 192,192,256,256,256,192 \
    --context-size 2 \
    --use-averaged-model true \
    --use-hat False
done
done
```




