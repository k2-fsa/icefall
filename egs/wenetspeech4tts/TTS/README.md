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

# Credits
- [vall-e](https://github.com/lifeiteng/vall-e)
