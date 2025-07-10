#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
from hifigan.config import v1, v2, v3
from hifigan.denoiser import Denoiser
from hifigan.models import Generator as HiFiGAN
from tokenizer import Tokenizer
from train import get_model, get_params
from tts_datamodule import LJSpeechTtsDataModule

from icefall.checkpoint import load_checkpoint
from icefall.utils import AttributeDict, setup_logger


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

    # The following arguments are used for inference on single text
    parser.add_argument(
        "--input-text",
        type=str,
        required=False,
        help="The text to generate speech for",
    )

    parser.add_argument(
        "--output-wav",
        type=str,
        required=False,
        help="The filename of the wave to save the generated speech",
    )

    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=22050,
        help="The sampling rate of the generated speech (default: 22050 for LJSpeech)",
    )

    return parser


def load_vocoder(checkpoint_path: Path) -> nn.Module:
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
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)["generator"]
    )
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def to_waveform(
    mel: torch.Tensor, vocoder: nn.Module, denoiser: nn.Module
) -> torch.Tensor:
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.squeeze()


def process_text(text: str, tokenizer: Tokenizer, device: str = "cpu") -> dict:
    x = tokenizer.texts_to_token_ids([text], add_sos=True, add_eos=True)
    x = torch.tensor(x, dtype=torch.long, device=device)
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    return {"x_orig": text, "x": x, "x_lengths": x_lengths}


def synthesize(
    model: nn.Module,
    tokenizer: Tokenizer,
    n_timesteps: int,
    text: str,
    length_scale: float,
    temperature: float,
    device: str = "cpu",
    spks=None,
) -> dict:
    text_processed = process_text(text=text, tokenizer=tokenizer, device=device)
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


def infer_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    vocoder: nn.Module,
    denoiser: nn.Module,
    tokenizer: Tokenizer,
) -> None:
    """Decode dataset.
    The ground-truth and generated audio pairs will be saved to `params.save_wav_dir`.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      tokenizer:
        Used to convert text to phonemes.
    """

    device = next(model.parameters()).device
    num_cuts = 0
    log_interval = 5

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    for batch_idx, batch in enumerate(dl):
        batch_size = len(batch["tokens"])

        texts = [c.supervisions[0].normalized_text for c in batch["cut"]]

        audio = batch["audio"]
        audio_lens = batch["audio_lens"].tolist()
        cut_ids = [cut.id for cut in batch["cut"]]

        for i in range(batch_size):
            output = synthesize(
                model=model,
                tokenizer=tokenizer,
                n_timesteps=params.n_timesteps,
                text=texts[i],
                length_scale=params.length_scale,
                temperature=params.temperature,
                device=device,
            )
            output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)

            sf.write(
                file=params.save_wav_dir / f"{cut_ids[i]}_pred.wav",
                data=output["waveform"],
                samplerate=params.data_args.sampling_rate,
                subtype="PCM_16",
            )
            sf.write(
                file=params.save_wav_dir / f"{cut_ids[i]}_gt.wav",
                data=audio[i].numpy(),
                samplerate=params.data_args.sampling_rate,
                subtype="PCM_16",
            )

        num_cuts += batch_size

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")


@torch.inference_mode()
def main():
    parser = get_parser()
    LJSpeechTtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.suffix = f"epoch-{params.epoch}"

    params.res_dir = params.exp_dir / "infer" / params.suffix
    params.save_wav_dir = params.res_dir / "wav"
    params.save_wav_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(f"{params.res_dir}/log-infer-{params.suffix}")
    logging.info("Infer started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

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

    # we need cut ids to organize tts results.
    args.return_cuts = True
    ljspeech = LJSpeechTtsDataModule(args)

    test_cuts = ljspeech.test_cuts()
    test_dl = ljspeech.test_dataloaders(test_cuts)

    if not Path(params.vocoder).is_file():
        raise ValueError(f"{params.vocoder} does not exist")

    vocoder = load_vocoder(params.vocoder)
    vocoder.to(device)

    denoiser = Denoiser(vocoder, mode="zeros")
    denoiser.to(device)

    if params.input_text is not None and params.output_wav is not None:
        logging.info("Synthesizing a single text")
        output = synthesize(
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
    else:
        logging.info("Decoding the test set")
        infer_dataset(
            dl=test_dl,
            params=params,
            model=model,
            vocoder=vocoder,
            denoiser=denoiser,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
