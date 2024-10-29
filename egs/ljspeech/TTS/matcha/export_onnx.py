#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script exports a Matcha-TTS model to ONNX.
Note that the model outputs fbank. You need to use a vocoder to convert
it to audio. See also ./export_onnx_hifigan.py
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
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


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, num_steps: int = 5):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        temperature: torch.Tensor,
        length_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args: :
          x: (batch_size, num_tokens), torch.int64
          x_lengths: (batch_size,), torch.int64
          temperature: (1,), torch.float32
          length_scale (1,), torch.float32
        Returns:
          audio: (batch_size, num_samples)

        """
        mel = self.model.synthesise(
            x=x,
            x_lengths=x_lengths,
            n_timesteps=self.num_steps,
            temperature=temperature,
            length_scale=length_scale,
        )["mel"]
        # mel: (batch_size, feat_dim, num_frames)

        return mel


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

    for num_steps in [2, 3, 4, 5, 6]:
        logging.info(f"num_steps: {num_steps}")
        wrapper = ModelWrapper(model, num_steps=num_steps)
        wrapper.eval()

        # Use a large value so the rotary position embedding in the text
        # encoder has a large initial length
        x = torch.ones(1, 1000, dtype=torch.int64)
        x_lengths = torch.tensor([x.shape[1]], dtype=torch.int64)
        temperature = torch.tensor([1.0])
        length_scale = torch.tensor([1.0])

        opset_version = 14
        filename = f"model-steps-{num_steps}.onnx"
        torch.onnx.export(
            wrapper,
            (x, x_lengths, temperature, length_scale),
            filename,
            opset_version=opset_version,
            input_names=["x", "x_length", "temperature", "length_scale"],
            output_names=["mel"],
            dynamic_axes={
                "x": {0: "N", 1: "L"},
                "x_length": {0: "N"},
                "mel": {0: "N", 2: "L"},
            },
        )

        meta_data = {
            "model_type": "matcha-tts",
            "language": "English",
            "voice": "en-us",
            "has_espeak": 1,
            "n_speakers": 1,
            "sample_rate": 22050,
            "version": 1,
            "model_author": "icefall",
            "maintainer": "k2-fsa",
            "dataset": "LJ Speech",
            "num_ode_steps": num_steps,
        }
        add_meta_data(filename=filename, meta_data=meta_data)
        print(meta_data)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
