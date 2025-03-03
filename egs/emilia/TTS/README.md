# Results
|        LLM Model                               | Flow matching Model | Seed-TTS test_zh CER  | Comment                                           |
|---------------------------------------|----------|-----------|--------|
| pretrained cosyvoice2 llm | pretrained cosyvoice2 unet |  1.45%   | See [paper](https://arxiv.org/abs/2412.10117)|
| pretrained cosyvoice2 llm | f5-tts-small (wenetspeech4tts) |  1.79% (16 steps)   | See [PR](https://github.com/k2-fsa/icefall/pull/1880)|
| llasa_cosyvoice2_token llm (Emilia 50k hours ZH) | f5-tts-small (wenetspeech4tts) |  1.89% (16 steps)   | |

# Introduction

[**Emilia**](https://huggingface.co/datasets/amphion/Emilia-Dataset) starts with over 101k
hours of speech across six languages, covering a wide range of speaking styles to enable more natural and spontaneous speech generation.

See https://arxiv.org/pdf/2407.05361.

> [!CAUTION]
> The next-gen Kaldi framework provides tools and models for generating high-quality, synthetic speech (Text-to-Speech, TTS).
> While these recipes has the potential to advance various fields such as accessibility, language education, and AI-driven solutions, it also carries certain ethical and legal responsibilities.
>
> By using this framework, you agree to the following:
> 1.	Legal and Ethical Use: You shall not use this framework, or any models derived from it, for any unlawful or unethical purposes. This includes, but is not limited to: Creating voice clones without the explicit, informed consent of the individual whose voice is being cloned. Engaging in any form of identity theft, impersonation, or fraud using cloned voices. Violating any local, national, or international laws regarding privacy, intellectual property, or personal data.
>
> 2.	Responsibility of Use: The users of this framework are solely responsible for ensuring that their use of voice cloning technologies complies with all applicable laws and ethical guidelines. We explicitly disclaim any liability for misuse of the technology.
>
> 3.	Attribution and Use of Open-Source Components: This project is provided under the Apache 2.0 license. Users must adhere to the terms of this license and provide appropriate attribution when required.
>
> 4.	No Warranty: This framework is provided “as-is,” without warranty of any kind, either express or implied. We do not guarantee that the use of this software will comply with legal requirements or that it will not infringe the rights of third parties.




# Llasa (cosyvoice2 token)

./llasa_cosyvoice2_token contains the code for training qwen2.5-0.5b models to predict cosyvoice2 semantic tokens.

Generated samples and training logs of [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) 50k hours Chinese data can be found [here](https://huggingface.co/yuekai/llasa_cosyvoice2_token_qwen_0.5b/tree/main).

Preparation:

```
# extract cosyvoice2 semantic tokens
bash prepare.sh --stage 3 --stop_stage 4

# Or you could use the prepared tokens.
huggingface-cli download yuekai/emilia_cosyvoice_v2_token --local-dir emilia_cosyvoice_v2_token
```

The training command is given below:

```
# docker: ghcr.io/swivid/f5-tts:main
# pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install -r llasa_cosyvoice2_token/requirements.txt
# pip install -r icefall/egs/wenetspeech4tts/TTS/f5-tts/requirements.txt

WANDB_KEY=$your_wandb_key
wandb login ${WANDB_KEY}
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen2.5-0.5B-Instruct
torchrun --nproc_per_node=8 train.py config.json
```

To inference with Icefall Emilia trained Chinese Llasa_cosyvoice2_token model, we need to use cosyvoice2 token flow matching [model](https://github.com/k2-fsa/icefall/pull/1880):
```
cd icefall/egs/wenetspeech4tts/TTS
huggingface-cli login
huggingface-cli download --local-dir ${exp_dir} yuekai/llasa_cosyvoice2_token_qwen_0.5b
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir bigvgan_v2_24khz_100band_256x
vocoder=./bigvgan_v2_24khz_100band_256x
split=test_zh
llm_path=llasa_cosyvoice2_token_qwen_0.5b/checkpoint-800000

huggingface-cli download --local-dir f5-tts-small-wenetspeech4tts-basic yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-10-avg-5.pt
torchrun --nproc_per_node=2 \
    f5-tts/infer_dist.py \
                --output_dir $output_dir \
                --batch_size 1 \
                --num_workers 2 \
                --llm-model-name-or-path $llm_path \
                --flow-matching-model-path $model_path \
                --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
                --use-cosyvoice-semantic-token True \
                --vocoder-dir $vocoder \
                --split-name $split -top-k 50 -top-p 0.95 -temperature 0.8 \
                --tokenizer-dir Qwen/Qwen2.5-0.5B-Instruct
# compute cer
huggingface-cli download yuekai/seed_tts_eval --local-dir seed_tts_eval --repo-type dataset
manifest=./seed_tts_eval/seedtts_testset/zh/meta.lst
bash local/compute_wer.sh $output_dir $manifest
```

# Credits
- [Llasa](https://arxiv.org/abs/2502.04128)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer/tree/main)
