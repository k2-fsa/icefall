#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation
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
Usage:
This script loads checkpoints and averages them.

(1) Average ZipVoice models before distill:
    python3 ./zipvoice/generate_averaged_model.py \
        --epoch 11 \
        --avg 4 \
        --distill 0 \
        --token-file data/tokens_emilia.txt \
        --exp-dir ./zipvoice/exp_zipvoice

    It will generate a file `epoch-11-avg-14.pt` in the given `exp_dir`.
    You can later load it by `torch.load("epoch-11-avg-4.pt")`.

(2) Average ZipVoice-Distill models (the first stage model):

    python3 ./zipvoice/generate_averaged_model.py \
        --iter 60000 \
        --avg 7 \
        --distill 1 \
        --token-file data/tokens_emilia.txt \
        --exp-dir ./zipvoice/exp_zipvoice_distill_1stage
"""

import argparse
from pathlib import Path

import torch
from model import get_distill_model, get_model
from tokenizer import TokenizerEmilia, TokenizerLibriTTS
from train_flow import add_model_arguments, get_params

from icefall.checkpoint import average_checkpoints_with_averaged_model, find_checkpoints
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=11,
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
        "'--epoch' or --iter",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipvoice/exp_zipvoice",
        help="The experiment dir",
    )

    parser.add_argument(
        "--distill",
        type=str2bool,
        default=False,
        help="Whether to use distill model. ",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="emilia",
        choices=["emilia", "libritts"],
        help="The used training dataset for the model to inference",
    )

    add_model_arguments(parser)

    return parser


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

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

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    print("Script started")

    params.device = torch.device("cpu")
    print(f"Device: {params.device}")

    print("About to create model")
    if params.distill:
        model = get_distill_model(params)
    else:
        model = get_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for" f" --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        print(
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
        print(
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
    if params.iter > 0:
        filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
    else:
        filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
    torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    print("Done!")


if __name__ == "__main__":
    main()
