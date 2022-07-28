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
./pruned_transducer_stateless3/export.py \
  --exp-dir ./pruned_transducer_stateless3/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10

It will generate a file exp_dir/pretrained.pt

To use the generated file with `pruned_transducer_stateless3/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./pruned_transducer_stateless3/decode.py \
        --exp-dir ./pruned_transducer_stateless3/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
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
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
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
        """,
    )

    parser.add_argument(
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit is ignored and it exports the model
        to onnx format. Three files will be generated:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
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


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

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

    if params.streaming_model:
        assert params.causal_convolution

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params, enable_giga=False)

    model.to(device)

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
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )
    elif params.avg == 1:
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
    opset_version = 11

    if params.onnx:
        logging.info("Exporting to onnx format")
        if True:
            x = torch.zeros(1, 100, 80, dtype=torch.float32)
            x_lens = torch.tensor([100], dtype=torch.int64)
            warmup = 1.0
            encoder_filename = params.exp_dir / "encoder.onnx"
            #  encoder_model = torch.jit.script(model.encoder)
            # It throws the following error for the above statement
            #
            # RuntimeError: Exporting the operator __is_ to ONNX opset version
            # 11 is not supported. Please feel free to request support or
            # submit a pull request on PyTorch GitHub.
            #
            # I cannot find which statement causes the above error.
            # torch.onnx.export() will use torch.jit.trace() internally, which
            # works well for the current reworked model

            encoder_model = model.encoder

            torch.onnx.export(
                encoder_model,
                (x, x_lens, warmup),
                encoder_filename,
                verbose=False,
                opset_version=opset_version,
                input_names=["x", "x_lens", "warmup"],
                output_names=["encoder_out", "encoder_out_lens"],
                dynamic_axes={
                    "x": {0: "N", 1: "T"},
                    "x_lens": {0: "N"},
                    "encoder_out": {0: "N", 1: "T"},
                    "encoder_out_lens": {0: "N"},
                },
            )
            logging.info(f"Saved to {encoder_filename}")

        if True:
            y = torch.zeros(10, 2, dtype=torch.int64)
            need_pad = False
            decoder_filename = params.exp_dir / "decoder.onnx"
            decoder_model = torch.jit.script(model.decoder)
            torch.onnx.export(
                decoder_model,
                (y, need_pad),
                decoder_filename,
                verbose=False,
                opset_version=opset_version,
                input_names=["y", "need_pad"],
                output_names=["decoder_out"],
                dynamic_axes={
                    "y": {0: "N", 1: "U"},
                    "decoder_out": {0: "N", 1: "U"},
                },
            )
            logging.info(f"Saved to {decoder_filename}")

        if True:
            encoder_out = torch.rand(1, 1, 3, 512, dtype=torch.float32)
            decoder_out = torch.rand(1, 1, 3, 512, dtype=torch.float32)
            project_input = False
            joiner_filename = params.exp_dir / "joiner.onnx"
            joiner_model = torch.jit.script(model.joiner)
            torch.onnx.export(
                joiner_model,
                (encoder_out, decoder_out, project_input),
                joiner_filename,
                verbose=False,
                opset_version=opset_version,
                input_names=["encoder_out", "decoder_out", "project_input"],
                output_names=["logit"],
                dynamic_axes={
                    "encoder_out": {0: "N", 1: "T", 2: "s_range"},
                    "decoder_out": {0: "N", 1: "T", 2: "s_range"},
                    "logit": {0: "N", 1: "T", 2: "s_range"},
                },
            )
            logging.info(f"Saved to {joiner_filename}")

        return

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
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
