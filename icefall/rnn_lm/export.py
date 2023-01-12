#!/usr/bin/env python3
#
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

import argparse
import logging
from pathlib import Path

import torch
from model import RnnLmModel

from icefall.checkpoint import load_checkpoint
from icefall.utils import AttributeDict, load_averaged_model, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=29,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the model",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=True,
        help="""True to save a model after applying torch.jit.script.
        """,
    )

    return parser


def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = AttributeDict({})
    params.update(vars(args))

    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = RnnLmModel(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        tie_weights=params.tie_weights,
    )

    model.to(device)

    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        model = load_averaged_model(
            params.exp_dir, model, params.epoch, params.avg, device
        )

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
