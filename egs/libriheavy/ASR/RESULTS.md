# Results

## zipformer (zipformer + pruned stateless transducer)

See <https://github.com/k2-fsa/icefall/pull/1261> for more details.

[zipformer](./zipformer)

### Non-streaming

#### Training on normalized text, i.e. Upper case without punctuation

##### normal-scaled model, number of model parameters: 65805511, i.e., 65.81 M

You can find a pretrained model, training logs at:
<https://www.modelscope.cn/models/pkufool/icefall-asr-zipformer-libriheavy-20230926/summary>

Note: The repository above contains three models trained on different subset of libriheavy exp(large set), exp_medium_subset(medium set),
exp_small_subset(small set).

Results of models:

| training set  |  decoding method    | librispeech clean | librispeech other | libriheavy clean | libriheavy other | comment            |
|---------------|---------------------|-------------------|-------------------|------------------|------------------|--------------------|
| small         |  greedy search      | 4.19              | 9.99              | 4.75             | 10.25            |--epoch 90 --avg 20 |
| small         | modified beam search| 4.05              | 9.89              | 4.68             | 10.01            |--epoch 90 --avg 20 |
| medium        |  greedy search      | 2.39              | 4.85              | 2.90             | 6.6              |--epoch 60 --avg 20 |
| medium        | modified beam search| 2.35              | 4.82              | 2.90             | 6.57             |--epoch 60 --avg 20 |
| large         |  greedy search      | 1.67              | 3.32              | 2.24             | 5.61             |--epoch 16 --avg 3  |
| large         | modified beam search| 1.62              | 3.36              | 2.20             | 5.57             |--epoch 16 --avg 3  |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./zipformer/train.py \
    --world-size 4 \
    --master-port 12365 \
    --exp-dir zipformer/exp \
    --num-epochs 60 \   # 16 for large; 90 for small
    --lr-hours 15000 \  # 20000 for large; 5000 for small
    --use-fp16 1 \
    --start-epoch 1 \
    --bpe-model data/lang_bpe_500/bpe.model \
    --max-duration 1000 \
    --subset medium
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search; do
  ./zipformer/decode.py \
      --epoch 16 \
      --avg 3 \
      --exp-dir zipformer/exp \
      --max-duration 1000 \
      --causal 0 \
      --decoding-method $m
done
```

#### Training on full formatted text, i.e. with casing and punctuation

##### normal-scaled model, number of model parameters: 66074067 , i.e., 66M

You can find a pretrained model, training logs at:
<https://www.modelscope.cn/models/pkufool/icefall-asr-zipformer-libriheavy-punc-20230830/summary>

Note: The repository above contains three models trained on different subset of libriheavy exp(large set), exp_medium_subset(medium set),
exp_small_subset(small set).

Results of models:

| training set  |  decoding method    | libriheavy clean (WER) | libriheavy other (WER) | libriheavy clean (CER) | libriheavy other (CER) | comment            |
|---------------|---------------------|-------------------|-------------------|------------------|------------------|--------------------|
| small         | modified beam search| 13.04             | 19.54             | 4.51             | 7.90             |--epoch 88 --avg 41 |
| medium        | modified beam search| 9.84              | 13.39             | 3.02             | 5.10             |--epoch 50 --avg 15 |
| large         | modified beam search| 7.76              | 11.32             | 2.41             | 4.22             |--epoch 16 --avg 2  |

The training command is:
```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

python ./zipformer/train.py \
    --world-size 4 \
    --master-port 12365 \
    --exp-dir zipformer/exp \
    --num-epochs 60 \   # 16 for large; 90 for small
    --lr-hours 15000 \  # 20000 for large; 10000 for small
    --use-fp16 1 \
    --train-with-punctuation 1 \
    --start-epoch 1 \
    --bpe-model data/lang_punc_bpe_756/bpe.model \
    --max-duration 1000 \
    --subset medium
```

The decoding command is:
```bash
export CUDA_VISIBLE_DEVICES="0"
for m in greedy_search modified_beam_search; do
  ./zipformer/decode.py \
      --epoch 16 \
      --avg 3 \
      --exp-dir zipformer/exp \
      --max-duration 1000 \
      --causal 0 \
      --decoding-method $m
done
```

## Zipformer PromptASR (zipformer + PromptASR + BERT text encoder)

#### [zipformer_prompt_asr](./zipformer_prompt_asr)

See <https://github.com/k2-fsa/icefall/pull/1250> for commit history and
our paper <https://arxiv.org/abs/2309.07414> for more details.



##### Training on the medium subset, with content & style prompt, **no** context list

You can find a pre-trained model, training logs, decoding logs, and decoding results at: <https://huggingface.co/marcoyang/icefall-promptasr-libriheavy-zipformer-BERT-2023-10-10>

The training command is:

```bash
causal=0
subset=medium
memory_dropout_rate=0.05
text_encoder_type=BERT

python ./zipformer_prompt_asr/train_bert_encoder.py \
    --world-size 4 \
    --start-epoch 1 \
    --num-epochs 60 \
    --exp-dir ./zipformer_prompt_asr/exp \
    --use-fp16 True \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration 1000 \
    --text-encoder-type $text_encoder_type \
    --text-encoder-dim 768 \
    --use-context-list 0 \
    --top-k $top_k \
    --use-style-prompt 1
```

The decoding results using utterance-level context (epoch-60-avg-10):

| decoding method      | lh-test-clean | lh-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   3.13        |    6.78       |    --use-pre-text False --use-style-prompt False      |
| modified_beam_search        |   2.86        |    5.93       |    --pre-text-transform upper-no-punc --style-text-transform upper-no-punc       |
| modified_beam_search        |   2.6        |    5.5       |    --pre-text-transform mixed-punc --style-text-transform mixed-punc       |


The decoding command is:

```bash
for style in mixed-punc upper-no-punc; do
    python ./zipformer_prompt_asr/decode_bert.py \
        --epoch 60 \
        --avg 10 \
        --use-averaged-model True \
        --post-normalization True \
        --causal False \
        --exp-dir ./zipformer_prompt_asr/exp \
        --manifest-dir data/fbank \
        --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
        --max-duration 1000 \
        --decoding-method modified_beam_search \
        --beam-size 4 \
        --text-encoder-type BERT \
        --text-encoder-dim 768 \
        --memory-layer 0 \
        --use-ls-test-set False \
        --use-ls-context-list False \
        --max-prompt-lens 1000 \
        --use-pre-text True \
        --use-style-prompt True \
        --style-text-transform $style \
        --pre-text-transform $style \
        --compute-CER 0
done
```

##### Training on the medium subset, with content & style prompt, **with** context list

You can find a pre-trained model, training logs, decoding logs, and decoding results at: <https://huggingface.co/marcoyang/icefall-promptasr-with-context-libriheavy-zipformer-BERT-2023-10-10>

This model is trained with an extra type of content prompt (context words), thus it does better
on **word-level** context biasing. Note that to train this model, please first run `prepare_prompt_asr.sh`
to prepare a manifest containing context words.

The training command is:

```bash

causal=0
subset=medium
memory_dropout_rate=0.05
text_encoder_type=BERT
use_context_list=True

# prepare the required data for context biasing
./prepare_prompt_asr.sh --stage 0 --stop_stage 1

python ./zipformer_prompt_asr/train_bert_encoder.py \
    --world-size 4 \
    --start-epoch 1 \
    --num-epochs 50 \
    --exp-dir ./zipformer_prompt_asr/exp \
    --use-fp16 True \
    --memory-dropout-rate $memory_dropout_rate \
    --causal $causal \
    --subset $subset \
    --manifest-dir data/fbank \
    --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
    --max-duration 1000 \
    --text-encoder-type $text_encoder_type \
    --text-encoder-dim 768 \
    --use-context-list $use_context_list \
    --top-k 10000 \
    --use-style-prompt 1
```

*Utterance-level biasing:*

| decoding method      | lh-test-clean | lh-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   3.17        |    6.72       |    --use-pre-text 0 --use-style-prompt 0    |
| modified_beam_search        |   2.91        |    6.24       |    --pre-text-transform upper-no-punc --style-text-transform upper-no-punc       |
| modified_beam_search        |   2.72        |    5.72       |    --pre-text-transform mixed-punc --style-text-transform mixed-punc       |


The decoding command for the table above is:

```bash
for style in mixed-punc upper-no-punc; do
    python ./zipformer_prompt_asr/decode_bert.py \
        --epoch 50 \
        --avg 10 \
        --use-averaged-model True \
        --post-normalization True \
        --causal False \
        --exp-dir ./zipformer_prompt_asr/exp \
        --manifest-dir data/fbank \
        --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
        --max-duration 1000 \
        --decoding-method modified_beam_search \
        --beam-size 4 \
        --text-encoder-type BERT \
        --text-encoder-dim 768 \
        --memory-layer 0 \
        --use-ls-test-set False \
        --use-ls-context-list False \
        --max-prompt-lens 1000 \
        --use-pre-text True \
        --use-style-prompt True \
        --style-text-transform $style \
        --pre-text-transform $style \
        --compute-CER 0
done
```

*Word-level biasing:*

The results are reported on LibriSpeech test-sets using the biasing list provided from <https://arxiv.org/abs/2104.02194>.
You need to set `--use-ls-test-set True` so that the LibriSpeech test sets are used.

| decoding method      | ls-test-clean | ls-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   2.4        |    5.08       |    --use-pre-text 0 --use-style-prompt 0    |
| modified_beam_search        |   2.14        |    4.62       |    --use-ls-context-list 1 --pre-text-transform mixed-punc --style-text-transform mixed-punc --ls-distractors 0       |
| modified_beam_search        |   2.14        |    4.64       |    --use-ls-context-list 1 --pre-text-transform mixed-punc --style-text-transform mixed-punc --ls-distractors 100       |

The decoding command is for the table above is:

```bash
use_ls_test_set=1
use_ls_context_list=1

for ls_distractors in 0 100; do
    python ./zipformer_prompt_asr/decode_bert.py \
        --epoch 50 \
        --avg 10 \
        --use-averaged-model True \
        --post-normalization True \
        --causal False \
        --exp-dir ./zipformer_prompt_asr/exp \
        --manifest-dir data/fbank \
        --bpe-model data/lang_bpe_500_fallback_coverage_0.99/bpe.model \
        --max-duration 1000 \
        --decoding-method modified_beam_search \
        --beam-size 4 \
        --text-encoder-type BERT \
        --text-encoder-dim 768 \
        --memory-layer 0 \
        --use-ls-test-set $use_ls_test_setse \
        --use-ls-context-list $use_ls_context_list \
        --ls-distractors $ls_distractors \
        --max-prompt-lens 1000 \
        --use-pre-text True \
        --use-style-prompt True \
        --style-text-transform mixed-punc \
        --pre-text-transform mixed-punc \
        --compute-CER 0
done

```
