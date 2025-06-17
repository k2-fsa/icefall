#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script generates speech with our pre-trained ZipVoice or
    ZipVoice-Distill models. Required models will be automatically
    downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:

export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice_distill" \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:
python3 zipvoice/zipvoice_infer.py \
    --model-name "zipvoice-distill" \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.
"""

import argparse
import datetime as dt
import os

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import torchaudio
from feature import TorchAudioFbank, TorchAudioFbankConfig
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from model import get_distill_model, get_model
from tokenizer import TokenizerEmilia
from utils import AttributeDict
from vocos import Vocos


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice_distill",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    add_model_arguments(parser)

    return parser


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--fm-decoder-downsampling-factor",
        type=str,
        default="1,2,4,2,1",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--fm-decoder-num-layers",
        type=str,
        default="2,2,4,4,4",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--fm-decoder-cnn-module-kernel",
        type=str,
        default="31,15,7,15,31",
        help="Sizes of convolutional kernels in convolution modules "
        "in each encoder stack: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--fm-decoder-feedforward-dim",
        type=int,
        default=1536,
        help="Feedforward dimension of the zipformer encoder layers, "
        "per stack, comma separated.",
    )

    parser.add_argument(
        "--fm-decoder-num-heads",
        type=int,
        default=4,
        help="Number of attention heads in the zipformer encoder layers: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--fm-decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in encoder stacks: a single int "
        "or comma-separated list.",
    )

    parser.add_argument(
        "--text-encoder-downsampling-factor",
        type=str,
        default="1",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--text-encoder-num-layers",
        type=str,
        default="4",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--text-encoder-feedforward-dim",
        type=int,
        default=512,
        help="Feedforward dimension of the zipformer encoder layers, "
        "per stack, comma separated.",
    )

    parser.add_argument(
        "--text-encoder-cnn-module-kernel",
        type=str,
        default="9",
        help="Sizes of convolutional kernels in convolution modules in "
        "each encoder stack: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--text-encoder-num-heads",
        type=int,
        default=4,
        help="Number of attention heads in the zipformer encoder layers: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--text-encoder-dim",
        type=int,
        default=192,
        help="Embedding dimension in encoder stacks: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=int,
        default=32,
        help="Query/key dimension per head in encoder stacks: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=int,
        default=12,
        help="Value dimension per head in encoder stacks: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=int,
        default=4,
        help="Positional-encoding dimension per head in encoder stacks: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default=48,
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension of timestamps embedding.",
    )

    parser.add_argument(
        "--text-embed-dim",
        type=int,
        default=192,
        help="Embedding dimension of text embedding.",
    )


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "sampling_rate": 24000,
            "frame_shift_ms": 256 / 24000 * 1000,
            "feat_dim": 100,
        }
    )

    return params


def get_vocoder():
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: nn.Module,
    vocoder: nn.Module,
    tokenizer: TokenizerEmilia,
    feature_extractor: TorchAudioFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """
    Generate waveform of a text based on a given prompt
        waveform and its transcription.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (nn.Module): The model used for generation.
        vocoder (nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (TokenizerEmilia): The tokenizer used to convert text to tokens.
        feature_extractor (TorchAudioFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Load and preprocess prompt wav
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)
    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics


def generate(
    res_dir: str,
    test_list: str,
    model: nn.Module,
    vocoder: nn.Module,
    tokenizer: TokenizerEmilia,
    feature_extractor: TorchAudioFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"
        metrics = generate_sentence(
            save_path=save_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=device,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
            target_rms=target_rms,
            feat_scale=feat_scale,
            sampling_rate=sampling_rate,
        )
        print(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    print(f"Average RTF: {np.sum(total_t)/np.sum(total_wav_seconds):.4f}")
    print(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder)/np.sum(total_wav_seconds):.4f}"
    )
    print(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder)/np.sum(total_wav_seconds):.4f}"
    )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) == parser.get_default(param):
            setattr(params, param, value)
            print(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    if torch.cuda.is_available():
        params.device = torch.device("cuda", 0)
    else:
        params.device = torch.device("cpu")

    token_file = hf_hub_download("zhu-han/ZipVoice", filename="tokens_emilia.txt")

    tokenizer = TokenizerEmilia(token_file)

    params.vocab_size = tokenizer.vocab_size
    params.pad_id = tokenizer.pad_id
    fix_random_seed(params.seed)

    if params.model_name == "zipvoice_distill":
        model = get_distill_model(params)
        model_ckpt = hf_hub_download(
            "zhu-han/ZipVoice", filename="exp_zipvoice_distill/model.safetensors"
        )
    else:
        model = get_model(params)
        model_ckpt = hf_hub_download(
            "zhu-han/ZipVoice", filename="exp_zipvoice/model.safetensors"
        )

    safetensors.torch.load_model(model, model_ckpt)

    model = model.to(params.device)
    model.eval()

    vocoder = get_vocoder()
    vocoder = vocoder.to(params.device)
    vocoder.eval()

    config = TorchAudioFbankConfig(
        sampling_rate=params.sampling_rate,
        n_mels=100,
        n_fft=1024,
        hop_length=256,
    )
    feature_extractor = TorchAudioFbank(config)

    if params.test_list:
        os.makedirs(params.res_dir, exist_ok=True)
        generate(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
        )
    else:
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
        )
    print("Done")


if __name__ == "__main__":
    main()
