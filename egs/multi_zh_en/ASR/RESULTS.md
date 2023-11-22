## Results

### Zh-En datasets bpe-based training results (Non-streaming) on Zipformer model

This is the [pull request #1238](https://github.com/k2-fsa/icefall/pull/1265) in icefall.

#### Non-streaming (Byte-Level BPE vocab_size=2000)

Best results (num of params : ~69M):

The training command:

```
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 35 \
  --use-fp16 1 \
  --max-duration 1000 \
  --num-workers 8
```

The decoding command:

```
for method in greedy_search modified_beam_search fast_beam_search; do
    ./zipformer/decode.py \
    --epoch 34 \
    --avg 19 \
    --decoding_method $method
done
```

Word Error Rates (WERs) listed below are produced by the checkpoint of the 20th epoch using greedy search and BPE model (# tokens is 2000).

|       Datasets       | TAL-CSASR | TAL-CSASR | 
|----------------------|-----------|-----------|
|   Zipformer WER (%)  |    dev    |   test    | 
|     greedy_search    |   6.65    |   6.69    |
| modified_beam_search |   6.46    |   6.51    |
|   fast_beam_search   |   6.57    |   6.68    |

Pre-trained model can be found here : https://huggingface.co/zrjin/icefall-asr-zipformer-multi-zh-en-2023-11-22, which is trained on LibriSpeech 960-hour training set (with speed perturbation), TAL-CSASR training set (with speed perturbation) and AiShell-2 (w/o speed perturbation).


