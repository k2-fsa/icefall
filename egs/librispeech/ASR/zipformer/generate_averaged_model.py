#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Yifan Yang)
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
(1) use the checkpoint exp_dir/epoch-xxx.pt
./zipformer/generate_averaged_model.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./zipformer/exp

It will generate a file `epoch-28-avg-15.pt` in the given `exp_dir`.
You can later load it by `torch.load("epoch-28-avg-15.pt")`.

(2) use the checkpoint exp_dir/checkpoint-iter.pt
./zipformer/generate_averaged_model.py \
    --iter 22000 \
    --avg 5 \
    --exp-dir ./zipformer/exp

It will generate a file `iter-22000-avg-5.pt` in the given `exp_dir`.
You can later load it by `torch.load("iter-22000-avg-5.pt")`.
"""


import argparse
from pathlib import Path

import k2
import torch
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import average_checkpoints_with_averaged_model, find_checkpoints


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
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
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens.txt",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
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

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    print("Script started")

    device = torch.device("cpu")
    print(f"Device: {device}")

    symbol_table = k2.SymbolTable.from_file(params.tokens)
    params.blank_id = symbol_table["<blk>"]
    params.unk_id = symbol_table["<unk>"]
    params.vocab_size = len(symbol_table)

    print("About to create model")
    model = get_model(params)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
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
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )
        filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
        torch.save({"model": model.state_dict()}, filename)
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
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )
        filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
        torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    print("Done!")


if __name__ == "__main__":
    main()
