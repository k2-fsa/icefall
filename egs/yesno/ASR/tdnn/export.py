#!/usr/bin/env python3

"""
This file is for exporting trained models to a checkpoint
or to a torchscript model.

(1) Generate the checkpoint tdnn/exp/pretrained.pt

./tdnn/export.py \
  --epoch 14 \
  --avg 2

See ./tdnn/pretrained.py for how to use the generated file.

(2) Generate torchscript model tdnn/exp/cpu_jit.pt

./tdnn/export.py \
  --epoch 14 \
  --avg 2 \
  --jit 1

See ./tdnn/jit_pretrained.py for how to use the generated file.
"""

import argparse
import logging

import torch
from model import Tdnn
from train import get_params

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=14,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=2,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
        """,
    )
    return parser


@torch.no_grad()
def main():
    args = get_parser().parse_args()

    params = get_params()
    params.update(vars(args))

    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)

    model = Tdnn(
        num_features=params.feature_dim,
        num_classes=max_token_id + 1,  # +1 for the blank symbol
    )
    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.load_state_dict(average_checkpoints(filenames))

    model.to("cpu")
    model.eval()

    if params.jit:
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
