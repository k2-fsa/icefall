#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang,
#                                            Yifan   Yang)
#           2023 NVIDIA Corporation (Author: Wen     Ding)
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

(1) Export to ONNX format

./pruned_transducer_stateless7_ctc_bs/export_onnx.py \
  --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 30 \
  --avg 13 \
  --onnx 1

It will generate the following files in the given `exp_dir`.
Check `onnx_check.py` for how to use them.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - joiner_encoder_proj.onnx
    - joiner_decoder_proj.onnx
    - lconv.onnx
    - frame_reducer.onnx
    - ctc_output.onnx

(2) Export to ONNX format which can be used in Triton Server
./pruned_transducer_stateless7_ctc_bs/export_onnx.py \
  --exp-dir ./pruned_transducer_stateless7_ctc_bs/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 30 \
  --avg 13 \
  --onnx-triton 1

It will generate the following files in the given `exp_dir`.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - joiner_encoder_proj.onnx
    - joiner_decoder_proj.onnx
    - lconv.onnx
    - ctc_output.onnx

Please see ./onnx_pretrained.py for usage of the generated files

Check
https://github.com/k2-fsa/sherpa-onnx
for how to use the exported models outside of icefall.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/yfyeung/icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29
    # You will find the pre-trained model in icefall-asr-librispeech-pruned_transducer_stateless7_ctc_bs-2023-01-29/exp
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import str2bool
from onnx_wrapper import TritonOnnxDecoder, TritonOnnxJoiner, TritonOnnxLconv


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
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_ctc_bs/exp",
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
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit is ignored and it exports the model
        to onnx format.
        It will generate the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
            - joiner_encoder_proj.onnx
            - joiner_decoder_proj.onnx
            - lconv.onnx
            - frame_reducer.onnx
            - ctc_output.onnx

        Refer to ./onnx_check.py and ./onnx_pretrained.py for how to use them.
        """,
    )
    parser.add_argument(
        "--onnx-triton",
        type=str2bool,
        default=False,
        help="""If True, and it exports the model
        to onnx format which can be used in NVIDIA triton server. 
        It will generate the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
            - joiner_encoder_proj.onnx
            - joiner_decoder_proj.onnx
            - lconv.onnx
            - ctc_output.onnx
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def export_encoder_model_onnx(
    encoder_model: nn.Module,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T, C)
        - encoder_out_lens, a tensor of shape (N,)

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    x = torch.zeros(15, 2000, 80, dtype=torch.float32)
    x_lens = torch.tensor([2000] * 15, dtype=torch.int64)

    #  encoder_model = torch.jit.script(encoder_model)
    # It throws the following error for the above statement
    #
    # RuntimeError: Exporting the operator __is_ to ONNX opset version
    # 11 is not supported. Please feel free to request support or
    # submit a pull request on PyTorch GitHub.
    #
    # I cannot find which statement causes the above error.
    # torch.onnx.export() will use torch.jit.trace() internally, which
    # works well for the current reworked model
    torch.onnx.export(
        encoder_model,
        (x, x_lens),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["encoder_out", "encoder_out_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "encoder_out": {0: "N", 1: "T"},
            "encoder_out_lens": {0: "N"},
        },
    )
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_onnx(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, 1, C)

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    y = torch.zeros(15, decoder_model.context_size, dtype=torch.int64)
    need_pad = False  # Always False, so we can use torch.jit.trace() here
    # Note(fangjun): torch.jit.trace() is more efficient than torch.jit.script()
    # in this case
    torch.onnx.export(
        decoder_model,
        (y, need_pad),
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y", "need_pad"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_filename}")


def export_decoder_model_onnx_triton(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX-Triton format.
    The exported model has one input:
        - y: a torch.int64 tensor of shape (N, decoder_model.context_size)
    and has one output:
        - decoder_out: a torch.float32 tensor of shape (N, 1, C)
    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)

    decoder_model = TritonOnnxDecoder(decoder_model)

    torch.onnx.export(
        decoder_model,
        (y),
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y"],
        output_names=["decoder_out"],
        dynamic_axes={
            "y": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:

        - projected_encoder_out: a tensor of shape (N, joiner_dim)
        - projected_decoder_out: a tensor of shape (N, joiner_dim)

    and produces one output:

        - logit: a tensor of shape (N, vocab_size)

    The exported encoder_proj model has one input:

        - encoder_out: a tensor of shape (N, encoder_out_dim)

    and produces one output:

        - projected_encoder_out: a tensor of shape (N, joiner_dim)

    The exported decoder_proj model has one input:

        - decoder_out: a tensor of shape (N, decoder_out_dim)

    and produces one output:

        - projected_decoder_out: a tensor of shape (N, joiner_dim)
    """
    encoder_proj_filename = str(joiner_filename).replace(".onnx", "_encoder_proj.onnx")
    decoder_proj_filename = str(joiner_filename).replace(".onnx", "_decoder_proj.onnx")

    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    joiner_dim = joiner_model.decoder_proj.weight.shape[0]

    projected_encoder_out = torch.rand(1, 1, 1, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(1, 1, 1, joiner_dim, dtype=torch.float32)

    project_input = False
    # Note: It uses torch.jit.trace() internally
    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out, project_input),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out",
            "decoder_out",
            "project_input",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")

    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.encoder_proj,
        encoder_out,
        encoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["projected_encoder_out"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "projected_encoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {encoder_proj_filename}")

    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.decoder_proj,
        decoder_out,
        decoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["decoder_out"],
        output_names=["projected_decoder_out"],
        dynamic_axes={
            "decoder_out": {0: "N"},
            "projected_decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_proj_filename}")


def export_joiner_model_onnx_triton(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:
        - projected_encoder_out: a tensor of shape (N, joiner_dim)
        - projected_decoder_out: a tensor of shape (N, joiner_dim)
    and produces one output:
        - logit: a tensor of shape (N, vocab_size)
    The exported encoder_proj model has one input:
        - encoder_out: a tensor of shape (N, encoder_out_dim)
    and produces one output:
        - projected_encoder_out: a tensor of shape (N, joiner_dim)
    The exported decoder_proj model has one input:
        - decoder_out: a tensor of shape (N, decoder_out_dim)
    and produces one output:
        - projected_decoder_out: a tensor of shape (N, joiner_dim)
    """
    encoder_proj_filename = str(joiner_filename).replace(".onnx", "_encoder_proj.onnx")
    decoder_proj_filename = str(joiner_filename).replace(".onnx", "_decoder_proj.onnx")

    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    joiner_dim = joiner_model.decoder_proj.weight.shape[0]

    projected_encoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)

    # Note: It uses torch.jit.trace() internally
    joiner_model = TritonOnnxJoiner(joiner_model)
    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out",
            "decoder_out",
            "project_input",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")

    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.encoder_proj,
        encoder_out,
        encoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["projected_encoder_out"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "projected_encoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {encoder_proj_filename}")

    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)
    torch.onnx.export(
        joiner_model.decoder_proj,
        decoder_out,
        decoder_proj_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["decoder_out"],
        output_names=["projected_decoder_out"],
        dynamic_axes={
            "decoder_out": {0: "N"},
            "projected_decoder_out": {0: "N"},
        },
    )
    logging.info(f"Saved to {decoder_proj_filename}")


def export_lconv_onnx(
    lconv: nn.Module,
    lconv_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the lconv to ONNX format.

    The exported lconv has two inputs:

        - lconv_input: a tensor of shape (N, T, C)
        - src_key_padding_mask: a tensor of shape (N, T)

    and has one output:

        - lconv_out: a tensor of shape (N, T, C)

    Args:
      lconv:
        The lconv to be exported.
      lconv_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    lconv_input = torch.zeros(15, 498, 384, dtype=torch.float32)
    src_key_padding_mask = torch.zeros(15, 498, dtype=torch.bool)

    torch.onnx.export(
        lconv,
        (lconv_input, src_key_padding_mask),
        lconv_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["lconv_input", "src_key_padding_mask"],
        output_names=["lconv_out"],
        dynamic_axes={
            "lconv_input": {0: "N", 1: "T"},
            "src_key_padding_mask": {0: "N", 1: "T"},
            "lconv_out": {0: "N", 1: "T"},
        },
    )
    logging.info(f"Saved to {lconv_filename}")


def export_lconv_onnx_triton(
    lconv: nn.Module,
    lconv_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the lconv to ONNX format.

    The exported lconv has two inputs:

        - lconv_input: a tensor of shape (N, T, C)
        - lconv_input_lens: a tensor of shape (N, )

    and has one output:

        - lconv_out: a tensor of shape (N, T, C)

    Args:
      lconv:
        The lconv to be exported.
      lconv_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    lconv_input = torch.zeros(15, 498, 384, dtype=torch.float32)
    lconv_input_lens = torch.tensor([498] * 15, dtype=torch.int64)

    lconv = TritonOnnxLconv(lconv)

    torch.onnx.export(
        lconv,
        (lconv_input, lconv_input_lens),
        lconv_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["lconv_input", "lconv_input_lens"],
        output_names=["lconv_out"],
        dynamic_axes={
            "lconv_input": {0: "N", 1: "T"},
            "lconv_input_lens": {0: "N"},
            "lconv_out": {0: "N", 1: "T"},
        },
    )
    logging.info(f"Saved to {lconv_filename}")


def export_frame_reducer_onnx(
    frame_reducer: nn.Module,
    frame_reducer_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the frame_reducer to ONNX format.

    The exported frame_reducer has four inputs:

        - x: a tensor of shape (N, T, C)
        - x_lens: a tensor of shape (N, T)
        - ctc_output: a tensor of shape (N, T, vocab_size)
        - blank_id: an int, always 0

    and has two outputs:

        - x_fr: a tensor of shape (N, T, C)
        - x_lens_fr: a tensor of shape (N, T)

    Args:
      frame_reducer:
        The frame_reducer to be exported.
      frame_reducer_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    x = torch.zeros(15, 498, 384, dtype=torch.float32)
    x_lens = torch.tensor([498] * 15, dtype=torch.int64)
    ctc_output = torch.randn(15, 498, 500, dtype=torch.float32)

    torch.onnx.export(
        frame_reducer,
        (x, x_lens, ctc_output),
        frame_reducer_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens", "ctc_output"],
        output_names=["out", "out_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "ctc_output": {0: "N", 1: "T"},
            "out": {0: "N", 1: "T"},
            "out_lens": {0: "N"},
        },
    )
    logging.info(f"Saved to {frame_reducer_filename}")


def export_ctc_output_onnx(
    ctc_output: nn.Module,
    ctc_output_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the frame_reducer to ONNX format.

    The exported frame_reducer has one inputs:

        - encoder_out: a tensor of shape (N, T, C)

    and has one output:

        - ctc_output: a tensor of shape (N, T, vocab_size)

    Args:
      ctc_output:
        The ctc_output to be exported.
      ctc_output_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    encoder_out = torch.zeros(15, 498, 384, dtype=torch.float32)

    torch.onnx.export(
        ctc_output,
        (encoder_out),
        ctc_output_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["ctc_output"],
        dynamic_axes={
            "encoder_out": {0: "N", 1: "T"},
            "ctc_output": {0: "N", 1: "T"},
        },
    )
    logging.info(f"Saved to {ctc_output_filename}")


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

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

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
            model.to(device)
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
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to("cpu")
    model.eval()

    convert_scaled_to_non_scaled(model, inplace=True)
    opset_version = 13
    logging.info("Exporting to onnx format")
    encoder_filename = params.exp_dir / "encoder.onnx"
    export_encoder_model_onnx(
        model.encoder,
        encoder_filename,
        opset_version=opset_version,
    )

    decoder_filename = params.exp_dir / "decoder.onnx"
    if params.onnx is True:
        export_decoder_model_onnx(
            model.decoder,
            decoder_filename,
            opset_version=opset_version,
        )
    elif params.onnx_triton is True:
        export_decoder_model_onnx_triton(
            model.decoder,
            decoder_filename,
            opset_version=opset_version,
        )

    joiner_filename = params.exp_dir / "joiner.onnx"
    if params.onnx is True:
        export_joiner_model_onnx(
            model.joiner,
            joiner_filename,
            opset_version=opset_version,
        )
    elif params.onnx_triton is True:
        export_joiner_model_onnx_triton(
            model.joiner,
            joiner_filename,
            opset_version=opset_version,
        )

    lconv_filename = params.exp_dir / "lconv.onnx"
    if params.onnx is True:
        export_lconv_onnx(
            model.lconv,
            lconv_filename,
            opset_version=opset_version,
        )
    elif params.onnx_triton is True:
        export_lconv_onnx_triton(
            model.lconv,
            lconv_filename,
            opset_version=opset_version,
        )

    if params.onnx is True:
        frame_reducer_filename = params.exp_dir / "frame_reducer.onnx"
        export_frame_reducer_onnx(
            model.frame_reducer,
            frame_reducer_filename,
            opset_version=opset_version,
        )

    ctc_output_filename = params.exp_dir / "ctc_output.onnx"
    export_ctc_output_onnx(
        model.ctc_output,
        ctc_output_filename,
        opset_version=opset_version,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
