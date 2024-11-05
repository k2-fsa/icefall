#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json
import logging
from pathlib import Path

import soundfile as sf
import torch
from hifigan.denoiser import Denoiser
from infer import load_vocoder, synthesise, to_waveform
from tokenizer import Tokenizer
from train import get_model, get_params

from icefall.checkpoint import load_checkpoint


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=4000,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        default="matcha/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--vocoder",
        type=Path,
        default="./generator_v1",
        help="Path to the vocoder",
    )

    parser.add_argument(
        "--tokens",
        type=Path,
        default="data/tokens.txt",
    )

    parser.add_argument(
        "--cmvn",
        type=str,
        default="data/fbank/cmvn.json",
        help="""Path to vocabulary.""",
    )

    parser.add_argument(
        "--input-text",
        type=str,
        required=True,
        help="The text to generate speech for",
    )

    parser.add_argument(
        "--output-wav",
        type=str,
        required=True,
        help="The filename of the wave to save the generated speech",
    )

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=22050,
        help="The sampling rate of the generated speech (default: 22050 for LJSpeech)",
    )

    return parser


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()

    params.update(vars(args))

    logging.info("Infer started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size
    params.model_args.n_vocab = params.vocab_size

    with open(params.cmvn) as f:
        stats = json.load(f)
        params.data_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.data_args.data_statistics.mel_std = stats["fbank_std"]

        params.model_args.data_statistics.mel_mean = stats["fbank_mean"]
        params.model_args.data_statistics.mel_std = stats["fbank_std"]

    # Number of ODE Solver steps
    params.n_timesteps = 2

    # Changes to the speaking rate
    params.length_scale = 1.0

    # Sampling temperature
    params.temperature = 0.667
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    model.to(device)
    model.eval()

    if not Path(params.vocoder).is_file():
        raise ValueError(f"{params.vocoder} does not exist")

    vocoder = load_vocoder(params.vocoder)
    vocoder.to(device)

    denoiser = Denoiser(vocoder, mode="zeros")
    denoiser.to(device)

    output = synthesise(
        model=model,
        tokenizer=tokenizer,
        n_timesteps=params.n_timesteps,
        text=params.input_text,
        length_scale=params.length_scale,
        temperature=params.temperature,
        device=device,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

    sf.write(
        file=params.output_wav,
        data=output["waveform"],
        samplerate=params.sampling_rate,
        subtype="PCM_16",
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
