# Results
|        Model                               | Seed-TTS test_zh CER  | Comment                                           |
|---------------------------------------|---------------------|--------|
| [vall-e](./valle)            | 4.33%    | ~150M |
| [f5-tts](./f5-tts)            | 3.02% (16 steps) / 2.42% (32 steps)    | F5-TTS-Small Config, ~155M |
| [f5-tts-semantic-token](./f5-tts) |  1.79% (16 steps)   | Using pretrained cosyvoice2 semantic tokens as inputs rather than text tokens, ~155M  |

# Introduction

[**WenetSpeech4TTS**](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS) is a multi-domain **Mandarin** corpus derived from the open-sourced [WenetSpeech](https://arxiv.org/abs/2110.03370) dataset.

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


# [VALL-E](https://arxiv.org/abs/2301.02111)

./valle contains the code for training VALL-E TTS model.

Checkpoints and training logs can be found [here](https://huggingface.co/yuekai/vall-e_wenetspeech4tts). The demo of the model trained with Wenetspeech4TTS Premium (945 hours) is available [here](https://huggingface.co/spaces/yuekai/valle_wenetspeech4tts_demo).

Preparation:

```
bash prepare.sh
```

The training command is given below:

```
world_size=8
exp_dir=exp/valle

## Train AR model
python3 valle/train.py --max-duration 320 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 1000 --valid-interval 2000 \
      --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir ${exp_dir} --world-size ${world_size}

## Train NAR model
# cd ${exp_dir}
# ln -s ${exp_dir}/best-valid-loss.pt epoch-99.pt  # --start-epoch 100=99+1
# cd -
python3 valle/train.py --max-duration 160 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 1000 --valid-interval 2000 \
      --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 100 --start-batch 0 --accumulate-grad-steps 2 \
      --exp-dir ${exp_dir} --world-size ${world_size}
```

To inference, use:
```
huggingface-cli login
huggingface-cli download --local-dir ${exp_dir} yuekai/vall-e_wenetspeech4tts
top_p=1.0
python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg}_top_p_${top_p} \
        --top-k -1 --temperature 1.0 \
        --text ./aishell3.txt \
        --checkpoint ${exp_dir}/epoch-${epoch}-avg-${avg}.pt \
        --text-extractor pypinyin_initials_finals --top-p ${top_p}
```

# [F5-TTS](https://arxiv.org/abs/2410.06885)

./f5-tts contains the code for training F5-TTS model.

Generated samples and training logs of wenetspeech basic 7k hours data can be found [here](https://huggingface.co/yuekai/f5-tts-small-wenetspeech4tts-basic/tensorboard).

Preparation:

```
bash prepare.sh --stage 5 --stop_stage 6
```
(Note: To compatiable with F5-TTS official checkpoint, we direclty use `vocab.txt` from [here.](https://github.com/SWivid/F5-TTS/blob/129014c5b43f135b0100d49a0c6804dd4cf673e1/data/Emilia_ZH_EN_pinyin/vocab.txt) To generate your own `vocab.txt`, you may refer to [the script](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/train/datasets/prepare_emilia.py).)

The training command is given below:

```
# docker: ghcr.io/swivid/f5-tts:main
# pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install kaldialign lhotse tensorboard bigvganinference sentencepiece

world_size=8
exp_dir=exp/f5-tts-small
python3 f5-tts/train.py --max-duration 700 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
      --base-lr 7.5e-5 --warmup-steps 20000 --num-epochs 60  \
      --num-decoder-layers 18 --nhead 12 --decoder-dim 768 \
      --exp-dir ${exp_dir} --world-size ${world_size}
```

To inference with Icefall Wenetspeech4TTS trained F5-Small, use:
```
huggingface-cli login
huggingface-cli download --local-dir seed_tts_eval yuekai/seed_tts_eval --repo-type dataset
huggingface-cli download --local-dir ${exp_dir} yuekai/f5-tts-small-wenetspeech4tts-basic
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir bigvgan_v2_24khz_100band_256x

manifest=./seed_tts_eval/seedtts_testset/zh/meta.lst
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-56-avg-14.pt
# skip
python3 f5-tts/generate_averaged_model.py \
    --epoch 56 \
    --avg 14 --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
    --exp-dir exp/f5_small


accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --manifest-file $manifest --output-dir $output_dir --decoder-dim 768 --nhead 12 --num-decoder-layers 18
bash local/compute_wer.sh $output_dir $manifest
```

To inference with official Emilia trained F5-Base, use:
```
huggingface-cli login
huggingface-cli download --local-dir seed_tts_eval yuekai/seed_tts_eval --repo-type dataset
huggingface-cli download --local-dir F5-TTS SWivid/F5-TTS
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir bigvgan_v2_24khz_100band_256x

manifest=./seed_tts_eval/seedtts_testset/zh/meta.lst
model_path=./F5-TTS/F5TTS_Base_bigvgan/model_1250000.pt

accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --manifest-file $manifest --output-dir $output_dir
bash local/compute_wer.sh $output_dir $manifest
```

# F5-TTS-Semantic-Token

./f5-tts contains the code for training F5-TTS-Semantic-Token. We replaced the text tokens in F5-TTS with pretrained cosyvoice2 semantic tokens. During inference, we use the pretrained CosyVoice2 LLM to predict the semantic tokens for target audios. We observed that this approach leads to faster convergence and improved prosody modeling results.

Generated samples and training logs of wenetspeech basic 7k hours data can be found [here](https://huggingface.co/yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic/tree/main).

Preparation:

```
# extract cosyvoice2 semantic tokens
bash prepare.sh --stage 5 --stop_stage 7
```

The training command is given below:

```
# docker: ghcr.io/swivid/f5-tts:main
# pip install k2==1.24.4.dev20241030+cuda12.4.torch2.4.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install kaldialign lhotse tensorboard bigvganinference sentencepiece

world_size=8
exp_dir=exp/f5-tts-semantic-token-small
python3 f5-tts/train.py --max-duration 700 --filter-min-duration 0.5 --filter-max-duration 20  \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 5000 --valid-interval 10000 \
      --base-lr 1e-4 --warmup-steps 20000 --average-period 0 \
      --num-epochs 10 --start-epoch 1 --start-batch 0 \
      --num-decoder-layers 18 --nhead 12 --decoder-dim 768 \
      --exp-dir ${exp_dir} --world-size ${world_size} \
      --decay-steps 600000 --prefix wenetspeech4tts_cosy_token --use-cosyvoice-semantic-token True
```

To inference with Icefall Wenetspeech4TTS trained F5-Small-Semantic-Token, use:
```
huggingface-cli login
huggingface-cli download --local-dir ${exp_dir} yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir bigvgan_v2_24khz_100band_256x

split=test_zh
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-10-avg-5.pt

accelerate launch f5-tts/infer.py --nfe 16 --model-path $model_path --split-name $split --output-dir $output_dir --decoder-dim 768 --nhead 12 --num-decoder-layers 18 --use-cosyvoice-semantic-token True
bash local/compute_wer.sh $output_dir $manifest
```

# Credits
- [VALL-E](https://github.com/lifeiteng/vall-e)
- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
