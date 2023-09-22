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

(1) Export to torchscript model using torch.jit.trace()

./conformer_ctc3/export.py \
  --exp-dir ./conformer_ctc3/exp \
  --lang-dir data/lang_bpe_500 \
  --epoch 20 \
  --avg 10 \
  --jit-trace 1

It will generates the file: `jit_trace.pt`.

(2) Export `model.state_dict()`

./conformer_ctc3/export.py \
  --exp-dir ./conformer_ctc3/exp \
  --lang-dir data/lang_bpe_500 \
  --epoch 20 \
  --avg 10

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

To use the generated file with `conformer_ctc3/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./conformer_ctc3/decode.py \
        --exp-dir ./conformer_ctc3/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 100 \
        --lang-dir data/lang_bpe_500
"""

import argparse
import logging
from pathlib import Path

import torch
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_ctc_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
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
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless4/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
        """,
    )

    parser.add_argument(
        "--streaming-model",
        type=str2bool,
        default=False,
        help="""Whether to export a streaming model, if the models in exp-dir
        are streaming model, this should be True.
        """,
    )

    add_model_arguments(parser)

    return parser


def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)
    num_classes = max_token_id + 1  # +1 for the blank
    params.vocab_size = num_classes

    if params.streaming_model:
        assert params.causal_convolution

    logging.info(params)

    logging.info("About to create model")
    model = get_ctc_model(params)

    model.to(device)

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
            logging.info(f"averaging {filenames}")
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
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to("cpu")
    model.eval()

    if params.jit_trace:
        # TODO: will support streaming mode
        assert not params.streaming_model
        convert_scaled_to_non_scaled(model, inplace=True)

        logging.info("Using torch.jit.trace()")

        x = torch.zeros(1, 100, 80, dtype=torch.float32)
        x_lens = torch.tensor([100], dtype=torch.int64)
        traced_model = torch.jit.trace(model, (x, x_lens))

        filename = params.exp_dir / "jit_trace.pt"
        traced_model.save(str(filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torch.jit.trace()")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
