## Results

### Zipformer PromptASR (zipformer + PromptASR + BERT text encoder)

#### [zipformer_prompt_asr](./zipformer_prompt_asr)

See <https://github.com/k2-fsa/icefall/pull/1250> for commit history and
our paper <https://arxiv.org/abs/2309.07414> for more details.



##### Training on the medium subset, with content & style prompt, no context list

You can find a pre-trained model, training logs, decoding logs, and decoding results at: <>

Number of model parameters:

| decoding method      | lh-test-clean | lh-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   2.64        |    5.55       |    --pre-text-transform mixed-punc --style-text-transform mixed-punc       |
| modified_beam_search        |   2.82        |    6.03       |    --pre-text-transform upper-no-punc --style-text-transform upper-no-punc       |
| modified_beam_search        |   2.64        |    5.55       |    --pre-text-transform mixed-punc --style-text-transform mixed-punc       |


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
    --use-context-list 0 \
    --top-k $top_k \
    --use-style-prompt 1
```

##### Training on the medium subset, with content & style prompt, with context list

You can find a pre-trained model, training logs, decoding logs, and decoding results at: <>

Number of model parameters:

*Utterance-level biasing:*

| decoding method      | lh-test-clean | lh-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   3.11        |    6.79       |    --use-pre-text 0 --use-style-prompt 0    |
| modified_beam_search        |   2.82        |    6.03       |    --pre-text-transform upper-no-punc --style-text-transform upper-no-punc       |
| modified_beam_search        |   2.64        |    5.55       |    --pre-text-transform mixed-punc --style-text-transform mixed-punc       |

*Word-level biasing:*

The results are reported on LibriSpeech test-sets using the biasing list provided from <https://arxiv.org/abs/2104.02194>. You need to set `--use-ls-test-set 1` for the following table.


| decoding method      | ls-test-clean | ls-test-other | comment             |
|----------------------|---------------|---------------|---------------------|
| modified_beam_search        |   2.69        |    5.28       |    --use-pre-text 0 --use-style-prompt 0    |
| modified_beam_search        |   2.32        |    4.77       |    --use-ls-context-list 1 --pre-text-transform mixed-punc --style-text-transform mixed-punc --ls-distractors 0       |
| modified_beam_search        |   2.36        |    4.91       |    --use-ls-context-list 1 --pre-text-transform mixed-punc --style-text-transform mixed-punc --ls-distractors 100       |



Note that to train this model, please first run `prepare_prompt_asr.sh` to prepare a
manifest containing context words.

The training command is:

```bash

causal=0
subset=medium
memory_dropout_rate=0.05
text_encoder_type=BERT

# prepare the required data for context biasing training & decoding
./prepare_prompt_asr.sh --stage 0 --stop_stage 1

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
    --use-context-list 1 \
    --top-k 10000 \
    --use-style-prompt 1
```
