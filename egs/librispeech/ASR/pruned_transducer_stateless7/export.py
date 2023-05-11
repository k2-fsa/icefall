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
<<<<<<< HEAD

Usage:

(1) Export to torchscript model using torch.jit.script()

./pruned_transducer_stateless7/export.py \
  --exp-dir ./pruned_transducer_stateless7/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 30 \
  --avg 9 \
  --jit 1

It will generate a file `cpu_jit.pt` in the given `exp_dir`. You can later
load it by `torch.jit.load("cpu_jit.pt")`.

Note `cpu` in the name `cpu_jit.pt` means the parameters when loaded into Python
are on CPU. You can use `to("cuda")` to move them to a CUDA device.

Check
https://github.com/k2-fsa/sherpa
for how to use the exported models outside of icefall.

(2) Export `model.state_dict()`

./pruned_transducer_stateless7/export.py \
  --exp-dir ./pruned_transducer_stateless7/exp \
=======
Usage:
./pruned_transducer_stateless5/export.py \
  --exp-dir ./pruned_transducer_stateless5/exp \
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10

<<<<<<< HEAD
It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

To use the generated file with `pruned_transducer_stateless7/decode.py`,
=======
It will generate a file exp_dir/pretrained.pt

To use the generated file with `pruned_transducer_stateless5/decode.py`,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
<<<<<<< HEAD
    ./pruned_transducer_stateless7/decode.py \
        --exp-dir ./pruned_transducer_stateless7/exp \
=======
    ./pruned_transducer_stateless5/decode.py \
        --exp-dir ./pruned_transducer_stateless5/exp \
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model
<<<<<<< HEAD

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11
    # You will find the pre-trained model in icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp
=======
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
<<<<<<< HEAD
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
=======
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
<<<<<<< HEAD
        default=30,
        help="""It specifies the checkpoint to use for decoding.
=======
        default=28,
        help="""It specifies the checkpoint to use for averaging.
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
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
<<<<<<< HEAD
        default=9,
=======
        default=15,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
<<<<<<< HEAD
        default=True,
=======
        default=False,
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
<<<<<<< HEAD
        default="pruned_transducer_stateless7/exp",
=======
        default="pruned_transducer_stateless5/exp",
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--jit",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.script.
<<<<<<< HEAD
        It will generate a file named cpu_jit.pt

        Check ./jit_pretrained.py for how to use it.
=======
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
<<<<<<< HEAD
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
=======
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    )

    add_model_arguments(parser)

    return parser


<<<<<<< HEAD
@torch.no_grad()
=======
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

<<<<<<< HEAD
=======
    assert args.jit is False, "Support torchscript will be added later"

>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

<<<<<<< HEAD
    model.to(device)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
=======
    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg]
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
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
    else:
        if params.iter > 0:
<<<<<<< HEAD
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
=======
            filenames = find_checkpoints(
                params.exp_dir, iteration=-params.iter
            )[: params.avg + 1]
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
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
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
<<<<<<< HEAD
=======
                    decompose=True
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
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
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
<<<<<<< HEAD
                )
            )

    model.to("cpu")
    model.eval()

    if params.jit is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        # We won't use the forward() method of the model in C++, so just ignore
        # it here.
        # Otherwise, one of its arguments is a ragged tensor and is not
        # torch scriptabe.
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)
=======
                    decompose=True
                )
            )

    model.eval()

    model.to("cpu")
    model.eval()

    if params.jit:
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        filename = params.exp_dir / "cpu_jit.pt"
        model.save(str(filename))
        logging.info(f"Saved to {filename}")
    else:
<<<<<<< HEAD
        logging.info("Not using torchscript. Export model.state_dict()")
=======
        logging.info("Not using torch.jit.script")
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
<<<<<<< HEAD
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
=======
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
>>>>>>> 1ab2a4c66231beb0ab0cc608bc27dba23fbd88a0

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
