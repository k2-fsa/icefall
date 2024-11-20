# Introduction

LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members.
The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus.
The main differences from the LibriSpeech corpus are listed below:
1. The audio files are at 24kHz sampling rate.
2. The speech is split at sentence breaks.
3. Both original and normalized texts are included.
4. Contextual information (e.g., neighbouring sentences) can be extracted.
5. Utterances with significant background noise are excluded.
For more information, refer to the paper "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019. If you use the LibriTTS corpus in your work, please cite this paper where it was introduced.

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


# VITS

This recipe provides a VITS model trained on the LibriTTS dataset.

Pretrained model can be found [here](https://huggingface.co/zrjin/icefall-tts-libritts-vits-2024-10-30).

The training command is given below:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./vits/train.py \
  --world-size 4 \
  --num-epochs 400 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir vits/exp \
  --max-duration 500
```

To inference, use:
```
./vits/infer.py \
  --exp-dir vits/exp \
  --epoch 400 \
  --tokens data/tokens.txt
```

# [VALL-E](https://arxiv.org/abs/2301.02111)

./valle contains the code for training VALL-E TTS model.

Checkpoints and training logs can be found [here](https://huggingface.co/yuekai/vall-e_libritts). The demo of the model trained with libritts and [libritts-r](https://www.openslr.org/141/) is available [here](https://huggingface.co/spaces/yuekai/valle-libritts-demo).

Preparation:

```
bash prepare.sh --start-stage 4
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
huggingface-cli download --local-dir ${exp_dir} yuekai/vall-e_libritts
top_p=1.0
python3 valle/infer.py --output-dir demos_epoch_${epoch}_avg_${avg}_top_p_${top_p} \
        --top-k -1 --temperature 1.0 \
        --text ./libritts.txt \
        --checkpoint ${exp_dir}/epoch-${epoch}-avg-${avg}.pt --top-p ${top_p}
```
