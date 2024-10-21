#!/usr/bin/env python3

import argparse
import datetime as dt
import logging
from pathlib import Path

import json
import numpy as np
import soundfile as sf
import torch
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from tokenizer import Tokenizer
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse
from tqdm.auto import tqdm
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
        default=2810,
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

    return parser


def load_vocoder(checkpoint_path):
    h = AttributeDict(v1)
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


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    np.save(folder / f"{filename}", output["mel"].cpu().numpy())
    sf.write(folder / f"{filename}.wav", output["waveform"], 22050, "PCM_24")


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
    print("output.shape", list(output.keys()), output["mel"].shape)
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
    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    model.eval()

    vocoder = load_vocoder("/star-fj/fangjun/open-source/Matcha-TTS/generator_v1")
    denoiser = Denoiser(vocoder, mode="zeros")

    texts = [
        "How are you doing? my friend.",
        "The Secret Service believed that it was very doubtful that any President would ride regularly in a vehicle with a fixed top, even though transparent.",
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.",
    ]

    # Number of ODE Solver steps
    n_timesteps = 3

    # Changes to the speaking rate
    length_scale = 1.0

    # Sampling temperature
    temperature = 0.667

    outputs, rtfs = [], []
    rtfs_w = []
    for i, text in enumerate(tqdm(texts)):
        output = synthesise(
            model=model,
            tokenizer=tokenizer,
            n_timesteps=n_timesteps,
            text=text,
            length_scale=length_scale,
            temperature=temperature,
        )  # , torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
        output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

        # Compute Real Time Factor (RTF) with HiFi-GAN
        t = (dt.datetime.now() - output["start_t"]).total_seconds()
        rtf_w = t * 22050 / (output["waveform"].shape[-1])

        # Pretty print
        print(f"{'*' * 53}")
        print(f"Input text - {i}")
        print(f"{'-' * 53}")
        print(output["x_orig"])
        print(f"{'*' * 53}")
        print(f"Phonetised text - {i}")
        print(f"{'-' * 53}")
        print(output["x"])
        print(f"{'*' * 53}")
        print(f"RTF:\t\t{output['rtf']:.6f}")
        print(f"RTF Waveform:\t{rtf_w:.6f}")
        rtfs.append(output["rtf"])
        rtfs_w.append(rtf_w)

        # Save the generated waveform
        save_to_folder(i, output, folder=f"./my-output-{params.epoch}")

    print(f"Number of ODE steps: {n_timesteps}")
    print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
    print(
        f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}"
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
