## Results

### MLS-English training results (Non-streaming) on zipformer model

#### Non-streaming

**WER on Test Set (Epoch 20)**

| Type          | Greedy | Beam search |
|---------------|--------|-------------|
| Non-streaming | 6.65   | 6.57        |


The training command:

```
./zipformer/train.py \
--world-size 8 \
--num-epochs 20 \
--start-epoch 9 \
--use-fp16 1 \
--exp-dir zipformer/exp \
--lang-dir data/lang/bpe_2000/
```

The decoding command:

```
./zipformer/decode.py \
    --epoch 20 \
    --exp-dir ./zipformer/exp \
    --lang-dir data/lang/bpe_2000/ \
    --decoding-method greedy_search
```


The pre-trained model is available here : [reazon-research/mls-english
](https://huggingface.co/reazon-research/mls-english)


Please note that this recipe was developed primarily as the source of English input in the bilingual Japanese-English recipe `multi_ja_en`, which uses ReazonSpeech and MLS English. 
