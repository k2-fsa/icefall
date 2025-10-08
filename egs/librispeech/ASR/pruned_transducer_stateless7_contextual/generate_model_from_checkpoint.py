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
(1) use the averaged model with checkpoint exp_dir/epoch-xxx.pt
./pruned_transducer_stateless7/generate_model_from_checkpoint.py \
    --epoch 28 \
    --avg 15 \
    --use-averaged-model True \
    --exp-dir ./pruned_transducer_stateless7/exp

It will generate a file `epoch-28-avg-15-use-averaged-model.pt` in the given `exp_dir`.
You can later load it by `torch.load("epoch-28-avg-15-use-averaged-model.pt")`.

(2) use the averaged model with checkpoint exp_dir/checkpoint-iter.pt
./pruned_transducer_stateless7/generate_model_from_checkpoint.py \
    --iter 22000 \
    --avg 5 \
    --use-averaged-model True \
    --exp-dir ./pruned_transducer_stateless7/exp

It will generate a file `iter-22000-avg-5-use-averaged-model.pt` in the given `exp_dir`.
You can later load it by `torch.load("iter-22000-avg-5-use-averaged-model.pt")`.

(3) use the original model with checkpoint exp_dir/epoch-xxx.pt
./pruned_transducer_stateless7/generate_model_from_checkpoint.py \
    --epoch 28 \
    --avg 15 \
    --use-averaged-model False \
    --exp-dir ./pruned_transducer_stateless7/exp

It will generate a file `epoch-28-avg-15.pt` in the given `exp_dir`.
You can later load it by `torch.load("epoch-28-avg-15.pt")`.

(4) use the original model with checkpoint exp_dir/checkpoint-iter.pt
./pruned_transducer_stateless7/generate_model_from_checkpoint.py \
    --iter 22000 \
    --avg 5 \
    --use-averaged-model False \
    --exp-dir ./pruned_transducer_stateless7/exp

It will generate a file `iter-22000-avg-5.pt` in the given `exp_dir`.
You can later load it by `torch.load("iter-22000-avg-5.pt")`.
"""


import argparse
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch

from train import add_model_arguments, get_params, get_transducer_model

from icefall.utils import str2bool
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)


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
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model."
        "If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
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

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    print("Script started")

    device = torch.device("cpu")
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    print("About to create model")
    model = get_transducer_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            print(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
            filename = params.exp_dir / f"iter-{params.iter}-avg-{params.avg}.pt"
            torch.save({"model": model.state_dict()}, filename)
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
            filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
            torch.save({"model": model.state_dict()}, filename)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            print(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
            filename = params.exp_dir / f"epoch-{params.epoch}-avg-{params.avg}.pt"
            torch.save({"model": model.state_dict()}, filename)
    else:
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
            filename = (
                params.exp_dir
                / f"iter-{params.iter}-avg-{params.avg}-use-averaged-model.pt"
            )
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
            filename = (
                params.exp_dir
                / f"epoch-{params.epoch}-avg-{params.avg}-use-averaged-model.pt"
            )
            torch.save({"model": model.state_dict()}, filename)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    print("Done!")


if __name__ == "__main__":
    main()
