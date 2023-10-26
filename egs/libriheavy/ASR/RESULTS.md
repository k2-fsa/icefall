## Results

### Zipformer PromptASR (zipformer + PromptASR + BERT text encoder)

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
