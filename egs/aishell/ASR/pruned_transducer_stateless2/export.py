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
./pruned_transducer_stateless2/export.py \
  --exp-dir ./pruned_transducer_stateless2/exp \
  --jit 0 \
  --epoch 29 \
  --avg 5

It will generate a file exp_dir/pretrained-epoch-29-avg-5.pt

To use the generated file with `pruned_transducer_stateless2/decode.py`,
you can do::

    cd /path/to/exp_dir
    ln -s pretrained-epoch-29-avg-5.pt epoch-9999.pt

    cd /path/to/egs/aishell/ASR
    ./pruned_transducer_stateless2/decode.py \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 100 \
        --lang-dir data/lang_char
"""

import argparse
import logging
from pathlib import Path

import torch
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=29,
        help="""It specifies the checkpoint to use for averaging.
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
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("pruned_transducer_stateless2/exp"),
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
        default=1,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


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

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if i >= 1:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

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
        filename = params.exp_dir / f"cpu_jit-epoch-{params.epoch}-avg-{params.avg}.pt"
        model.save(str(filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torch.jit.script")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = (
            params.exp_dir / f"pretrained-epoch-{params.epoch}-avg-{params.avg}.pt"
        )
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
