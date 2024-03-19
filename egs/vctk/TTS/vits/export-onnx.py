#!/usr/bin/env python3
#
# Copyright   2023-2024  Xiaomi Corporation     (Author: Zengwei Yao,
#                                                        Zengrui Jin,)
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
This script exports a VITS model from PyTorch to ONNX.

Export the model to ONNX:
./vits/export-onnx.py \
  --epoch 1000 \
  --exp-dir vits/exp \
  --tokens data/tokens.txt

It will generate two files inside vits/exp:
  - vits-epoch-1000.onnx
  - vits-epoch-1000.int8.onnx (quantizated model)

See ./test_onnx.py for how to use the exported ONNX models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
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
        default=1000,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="vits/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--speakers",
        type=Path,
        default=Path("data/speakers.txt"),
        help="Path to speakers.txt file.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="data/tokens.txt",
        help="""Path to vocabulary.""",
    )

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxModel(nn.Module):
    """A wrapper for VITS generator."""

    def __init__(self, model: nn.Module):
        """
        Args:
          model:
            A VITS generator.
          frame_shift:
            The frame shift in samples.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
        noise_scale: float = 0.667,
        alpha: float = 1.0,
        noise_scale_dur: float = 0.8,
        speaker: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Please see the help information of VITS.inference_batch

        Args:
          tokens:
            Input text token indexes (1, T_text)
          tokens_lens:
            Number of tokens of shape (1,)
          noise_scale (float):
            Noise scale parameter for flow.
          noise_scale_dur (float):
            Noise scale parameter for duration predictor.
          speaker (int):
            Speaker ID.
          alpha (float):
            Alpha parameter to control the speed of generated speech.

        Returns:
          Return a tuple containing:
            - audio, generated wavform tensor, (B, T_wav)
        """
        audio, _, _ = self.model.inference(
            text=tokens,
            text_lengths=tokens_lens,
            noise_scale=noise_scale,
            noise_scale_dur=noise_scale_dur,
            sids=speaker,
            alpha=alpha,
        )
        return audio


def export_model_onnx(
    model: nn.Module,
    model_filename: str,
    vocab_size: int,
    n_speakers: int,
    opset_version: int = 11,
) -> None:
    """Export the given generator model to ONNX format.
    The exported model has one input:

        - tokens, a tensor of shape (1, T_text); dtype is torch.int64

    and it has one output:

        - audio, a tensor of shape (1, T'); dtype is torch.float32

    Args:
      model:
        The VITS generator.
      model_filename:
        The filename to save the exported ONNX model.
      vocab_size:
        Number of tokens used in training.
      opset_version:
        The opset version to use.
    """
    tokens = torch.randint(low=0, high=vocab_size, size=(1, 13), dtype=torch.int64)
    tokens_lens = torch.tensor([tokens.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_dur = torch.tensor([1], dtype=torch.float32)
    alpha = torch.tensor([1], dtype=torch.float32)
    speaker = torch.tensor([1], dtype=torch.int64)

    torch.onnx.export(
        model,
        (tokens, tokens_lens, noise_scale, alpha, noise_scale_dur, speaker),
        model_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "tokens_lens",
            "noise_scale",
            "alpha",
            "noise_scale_dur",
            "speaker",
        ],
        output_names=["audio"],
        dynamic_axes={
            "tokens": {0: "N", 1: "T"},
            "tokens_lens": {0: "N"},
            "audio": {0: "N", 1: "T"},
            "speaker": {0: "N"},
        },
    )

    meta_data = {
        "model_type": "vits",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "icefall",  # must be icefall for models from icefall
        "language": "English",
        "voice": "en-us",  # Choose your language appropriately
        "has_espeak": 1,
        "n_speakers": n_speakers,
        "sample_rate": 22050,  # Must match the real sample rate
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=model_filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    tokenizer = Tokenizer(params.tokens)
    params.blank_id = tokenizer.pad_id
    params.vocab_size = tokenizer.vocab_size

    with open(args.speakers) as f:
        speaker_map = {line.strip(): i for i, line in enumerate(f)}
    params.num_spks = len(speaker_map)

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)

    model = model.generator
    model.to("cpu")
    model.eval()

    model = OnnxModel(model=model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"generator parameters: {num_param}")

    suffix = f"epoch-{params.epoch}"

    opset_version = 13

    logging.info("Exporting encoder")
    model_filename = params.exp_dir / f"vits-{suffix}.onnx"
    export_model_onnx(
        model,
        model_filename,
        params.vocab_size,
        params.num_spks,
        opset_version=opset_version,
    )
    logging.info(f"Exported generator to {model_filename}")

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    logging.info("Generate int8 quantization models")

    model_filename_int8 = params.exp_dir / f"vits-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=model_filename,
        model_output=model_filename_int8,
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
