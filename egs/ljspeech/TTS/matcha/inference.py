#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import soundfile as sf
import torch
from matcha.hifigan.config import v1, v2, v3
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.models import Generator as HiFiGAN
from tokenizer import Tokenizer
from train import get_model, get_params

from icefall.checkpoint import load_checkpoint
from icefall.utils import AttributeDict


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
        default="matcha/exp-new-3",
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

    return parser


def load_vocoder(checkpoint_path):
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith("v1"):
        h = AttributeDict(v1)
    elif checkpoint_path.endswith("v2"):
        h = AttributeDict(v2)
    elif checkpoint_path.endswith("v3"):
        h = AttributeDict(v3)
    else:
        raise ValueError(f"supports only v1, v2, and v3, given {checkpoint_path}")

    hifigan = HiFiGAN(h).to("cpu")
    hifigan.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")["generator"]
    )
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def to_waveform(mel, vocoder, denoiser):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()


def process_text(text: str, tokenizer):
    x = tokenizer.texts_to_token_ids([text], add_sos=True, add_eos=True)
    x = torch.tensor(x, dtype=torch.long)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")
    return {"x_orig": text, "x": x, "x_lengths": x_lengths}


def synthesise(
    model, tokenizer, n_timesteps, text, length_scale, temperature, spks=None
):
    text_processed = process_text(text, tokenizer)
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spks,
        length_scale=length_scale,
    )
    # merge everything to one dict
    output.update({"start_t": start_t, **text_processed})
    return output


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = get_params()

    params.update(vars(args))

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
    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not Path(f"{params.exp_dir}/epoch-{params.epoch}.pt").is_file():
        raise ValueError("{params.exp_dir}/epoch-{params.epoch}.pt does not exist")

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    model.eval()

    if not Path(params.vocoder).is_file():
        raise ValueError(f"{params.vocoder} does not exist")

    vocoder = load_vocoder(params.vocoder)
    denoiser = Denoiser(vocoder, mode="zeros")

    # Number of ODE Solver steps
    n_timesteps = 2

    # Changes to the speaking rate
    length_scale = 1.0

    # Sampling temperature
    temperature = 0.667

    output = synthesise(
        model=model,
        tokenizer=tokenizer,
        n_timesteps=n_timesteps,
        text=params.input_text,
        length_scale=length_scale,
        temperature=temperature,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

    sf.write(params.output_wav, output["waveform"], 22050, "PCM_16")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
