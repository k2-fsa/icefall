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
