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
    --decoding-method $method
done
```

Word Error Rates (WERs) listed below are produced by the checkpoint of the 20th epoch using greedy search and BPE model (# tokens is 2000).

|       Datasets       | TAL-CSASR | TAL-CSASR | AiShell-2 | AiShell-2 | LibriSpeech | LibriSpeech | 
|----------------------|-----------|-----------|-----------|-----------|-------------|-------------|
|   Zipformer WER (%)  |    dev    |   test    |    dev    |   test    |  test-clean |  test-other | 
|     greedy_search    |   6.65    |   6.69    |    6.57   |   7.03    |    2.43     |    5.70     |
| modified_beam_search |   6.46    |   6.51    |    6.18   |   6.60    |    2.41     |    5.57     |
|   fast_beam_search   |   6.57    |   6.68    |    6.40   |   6.74    |    2.40     |    5.56     |

Pre-trained model can be found here : https://huggingface.co/zrjin/icefall-asr-zipformer-multi-zh-en-2023-11-22, which is trained on LibriSpeech 960-hour training set (with speed perturbation), TAL-CSASR training set (with speed perturbation) and AiShell-2 (w/o speed perturbation).


