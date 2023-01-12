#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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

# This script converts several saved checkpoints
# to a single one using model averaging.
"""
Usage:
./transducer_stateless_modified/export.py \
  --exp-dir ./transducer_stateless_modified/exp \
  --epoch 64 \
  --avg 33

It will generate a file exp_dir/pretrained.pt

To use the generated file with `transducer_stateless_modified/decode.py`,
you can do::

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/aishell/ASR
    ./transducer_stateless_modified/decode.py \
        --exp-dir ./transducer_stateless_modified/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 100 \
        --lang-dir data/lang_char
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from conformer import Conformer
from decoder import Decoder
from joiner import Joiner
from model import Transducer

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.utils import AttributeDict, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=10,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("transducer_stateless_modified/exp"),
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=Path("data/lang_char"),
        help="The lang dir",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "feature_dim": 80,
            "encoder_out_dim": 512,
            "subsampling_factor": 4,
            "attention_dim": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "num_encoder_layers": 12,
            "vgg_frontend": False,
            "env_info": get_env_info(),
        }
    )
    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.encoder_out_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.attention_dim,
        nhead=params.nhead,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        vgg_frontend=params.vgg_frontend,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.encoder_out_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        input_dim=params.encoder_out_dim,
        output_dim=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
    )
    return model


def main():
    args = get_parser().parse_args()

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    lexicon = Lexicon(params.lang_dir)

    params.blank_id = 0
    params.vocab_size = max(lexicon.tokens) + 1

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    model.to(device)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to("cpu")
    model.eval()

    if params.jit:
        # We won't use the forward() method of the model in C++, so just ignore
        # it here.
        # Otherwise, one of its arguments is a ragged tensor and is not
        # torch scriptabe.
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)
        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        filename = params.exp_dir / "cpu_jit.pt"
        model.save(str(filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torch.jit.script")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
