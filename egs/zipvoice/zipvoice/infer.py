#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Wei Kang
#                                                       Han Zhu)
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
This script loads checkpoints to generate waveforms.
This script is supposed to be used with the model trained by yourself.
If you want to use the pre-trained checkpoints provided by us, please refer to zipvoice_infer.py.

(1) Usage with a pre-trained checkpoint:

    (a) ZipVoice model before distill:
        python3 zipvoice/infer.py \
            --checkpoint zipvoice/exp_zipvoice/epoch-11-avg-4.pt \
            --distill 0 \
            --token-file "data/tokens_emilia.txt" \
            --test-list test.tsv \
            --res-dir results/test \
            --num-step 16 \
            --guidance-scale 1

    (b) ZipVoice-Distill:
        python3 zipvoice/infer.py \
            --checkpoint zipvoice/exp_zipvoice_distill/checkpoint-2000.pt \
            --distill 1 \
            --token-file "data/tokens_emilia.txt" \
            --test-list test.tsv \
            --res-dir results/test_distill \
            --num-step 8 \
            --guidance-scale 3

(2) Usage with a directory of checkpoints (may requires checkpoint averaging):

    (a) ZipVoice model before distill:
        python3 flow_match/infer.py \
            --exp-dir zipvoice/exp_zipvoice \
            --epoch 11 \
            --avg 4 \
            --distill 0 \
            --token-file "data/tokens_emilia.txt" \
            --test-list test.tsv \
            --res-dir results \
            --num-step 16 \
            --guidance-scale 1

    (b) ZipVoice-Distill:
        python3 flow_match/infer.py \
            --exp-dir zipvoice/exp_zipvoice_distill/ \
            --iter 2000 \
            --avg 0 \
            --distill 1 \
            --token-file "data/tokens_emilia.txt" \
            --test-list test.tsv \
            --res-dir results \
            --num-step 8 \
            --guidance-scale 3
"""


import argparse
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from checkpoint import load_checkpoint
from feature import TorchAudioFbank, TorchAudioFbankConfig
from lhotse.utils import fix_random_seed
from model import get_distill_model, get_model
from tokenizer import TokenizerEmilia, TokenizerLibriTTS
from train_flow import add_model_arguments, get_params
from vocos import Vocos

from icefall.checkpoint import average_checkpoints_with_averaged_model, find_checkpoints
from icefall.utils import AttributeDict, setup_logger, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint for inference. "
        "If it is None, will use checkpoints under exp_dir",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipvoice/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=4,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' or '--iter', avg=0 means no avg",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="The local vocos vocoder path, downloaded from huggingface, "
        "will download the vocodoer from huggingface if it is None.",
    )

    parser.add_argument(
        "--distill",
        type=str2bool,
        default=False,
        help="Whether it is a distilled TTS model.",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesize in the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="Path name of the generated wavs dir",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts"],
        help="The used training dataset for the model to inference",
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=16,
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


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocos_local_path = "model/huggingface/vocos-mel-24khz/"
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
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
    wav = wav[0].cpu().numpy()
    sf.write(save_path, wav, sampling_rate)

    return metrics


def generate(
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    tokenizer: TokenizerEmilia,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    config = TorchAudioFbankConfig(
        sampling_rate=params.sampling_rate,
        n_mels=100,
        n_fft=1024,
        hop_length=256,
    )
    feature_extractor = TorchAudioFbank(config)

    with open(params.test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{params.wav_dir}/{wav_name}.wav"
        metrics = generate_sentence(
            save_path=save_path,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            text=text,
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
        print(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    print(f"Average RTF: " f"{np.sum(total_t)/np.sum(total_wav_seconds):.4f}")
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
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    if params.iter > 0:
        params.suffix = (
            f"wavs-iter-{params.iter}-avg"
            f"-{params.avg}-step-{params.num_step}-scale-{params.guidance_scale}"
        )
    elif params.epoch > 0:
        params.suffix = (
            f"wavs-epoch-{params.epoch}-avg"
            f"-{params.avg}-step-{params.num_step}-scale-{params.guidance_scale}"
        )
    else:
        params.suffix = "wavs"

    setup_logger(f"{params.res_dir}/log-infer-{params.suffix}")
    logging.info("Decoding started")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", 0)
    else:
        params.device = torch.device("cpu")

    logging.info(f"Device: {params.device}")

    if params.dataset == "emilia":
        tokenizer = TokenizerEmilia(
            token_file=params.token_file, token_type=params.token_type
        )
    elif params.dataset == "libritts":
        tokenizer = TokenizerLibriTTS(
            token_file=params.token_file, token_type=params.token_type
        )

    params.vocab_size = tokenizer.vocab_size
    params.pad_id = tokenizer.pad_id

    logging.info(params)
    fix_random_seed(params.seed)

    logging.info("About to create model")
    if params.distill:
        model = get_distill_model(params)
    else:
        model = get_model(params)

    if params.checkpoint:
        load_checkpoint(params.checkpoint, model, strict=True)
    else:
        if params.avg == 0:
            if params.iter > 0:
                load_checkpoint(
                    f"{params.exp_dir}/checkpoint-{params.iter}.pt", model, strict=True
                )
            else:
                load_checkpoint(
                    f"{params.exp_dir}/epoch-{params.epoch}.pt", model, strict=True
                )
        else:
            if params.iter > 0:
                filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                    : params.avg + 1
                ]
                if len(filenames) == 0:
                    raise ValueError(
                        f"No checkpoints found for"
                        f" --iter {params.iter}, --avg {params.avg}"
                    )
                elif len(filenames) < params.avg + 1:
                    raise ValueError(
                        f"Not enough checkpoints ({len(filenames)}) found for"
                        f" --iter {params.iter}, --avg {params.avg}"
                    )
                filename_start = filenames[-1]
                filename_end = filenames[0]
                logging.info(
                    "Calculating the averaged model over iteration checkpoints"
                    f" from {filename_start} (excluded) to {filename_end}"
                )
                model.to(params.device)
                model.load_state_dict(
                    average_checkpoints_with_averaged_model(
                        filename_start=filename_start,
                        filename_end=filename_end,
                        device=params.device,
                    ),
                    strict=True,
                )
            else:
                assert params.avg > 0, params.avg
                start = params.epoch - params.avg
                assert start >= 1, start
                filename_start = f"{params.exp_dir}/epoch-{start}.pt"
                filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
                logging.info(
                    f"Calculating the averaged model over epoch range from "
                    f"{start} (excluded) to {params.epoch}"
                )
                model.to(params.device)
                model.load_state_dict(
                    average_checkpoints_with_averaged_model(
                        filename_start=filename_start,
                        filename_end=filename_end,
                        device=params.device,
                    ),
                    strict=True,
                )

    model = model.to(params.device)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(params.device)
    vocoder.eval()
    num_param = sum([p.numel() for p in vocoder.parameters()])
    logging.info(f"Number of vocoder parameters: {num_param}")

    params.wav_dir = f"{params.res_dir}/{params.suffix}"
    os.makedirs(params.wav_dir, exist_ok=True)

    assert (
        params.test_list is not None
    ), "Please provide --test-list for speech synthesize."
    generate(
        params=params,
        model=model,
        vocoder=vocoder,
        tokenizer=tokenizer,
    )

    logging.info("Done!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
