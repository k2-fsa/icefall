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
# to one using model averaging.

import argparse
import logging
from pathlib import Path

import torch
from conformer import Conformer

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.utils import str2bool, AttributeDict
from icefall.lexicon import Lexicon


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=34,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=20,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe",
        help="""It contains language related input files such as "lexicon.txt"
        """,
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=True,
        help="""True to save a model after using torch.jit.script.
        """,
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "a": 1,
            "b": 10,
            "feature_dim": 80,
            "subsampling_factor": 4,
            "use_feat_batchnorm": True,
            "attention_dim": 512,
            "nhead": 8,
            "num_decoder_layers": 6,
        }
    )
    return params


def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)

    params = get_params()
    params.update(vars(args))

    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.attention_dim,
        num_classes=num_classes,
        subsampling_factor=params.subsampling_factor,
        num_decoder_layers=params.num_decoder_layers,
        vgg_frontend=False,
        use_feat_batchnorm=params.use_feat_batchnorm,
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

    if params.jit:
        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        model.save(f"{params.exp_dir}/cpu_jit.pt")
    else:
        logging.info("Not using torch.jit.script")
        torch.save(
            {"model": model.state_dict()}, f"{params.exp_dir}/pretrained.pt"
        )


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
