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

(1) Export to torchscript model using torch.jit.script()

./pruned_transducer_stateless3/export.py \
  --exp-dir ./pruned_transducer_stateless3/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --jit 1

It will generate a file `cpu_jit.pt` in the given `exp_dir`. You can later
load it by `torch.jit.load("cpu_jit.pt")`.

Note `cpu` in the name `cpu_jit.pt` means the parameters when loaded into Python
are on CPU. You can use `to("cuda")` to move them to a CUDA device.

It will also generate 3 other files: `encoder_jit_script.pt`,
`decoder_jit_script.pt`, and `joiner_jit_script.pt`.

(2) Export to torchscript model using torch.jit.trace()

./pruned_transducer_stateless3/export.py \
  --exp-dir ./pruned_transducer_stateless3/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --jit-trace 1

It will generates 3 files: `encoder_jit_trace.pt`,
`decoder_jit_trace.pt`, and `joiner_jit_trace.pt`.


(3) Export to ONNX format

./pruned_transducer_stateless3/export.py \
  --exp-dir ./pruned_transducer_stateless3/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10 \
  --onnx 1

It will generate the following files in the given `exp_dir`.
Check `onnx_check.py` for how to use them.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - joiner_encoder_proj.onnx
    - joiner_decoder_proj.onnx

Please see ./onnx_pretrained.py for usage of the generated files

Check
https://github.com/k2-fsa/sherpa-onnx
for how to use the exported models outside of icefall.

(4) Export `model.state_dict()`

./pruned_transducer_stateless3/export.py \
  --exp-dir ./pruned_transducer_stateless3/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

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

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13
    # You will find the pre-trained model in icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/exp
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
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
        It will generate 4 files:
         - encoder_jit_script.pt
         - decoder_jit_script.pt
         - joiner_jit_script.pt
         - cpu_jit.pt (which combines the above 3 files)

        Check ./jit_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--jit-trace",
        type=str2bool,
        default=False,
        help="""True to save a model after applying torch.jit.trace.
        It will generate 3 files:
         - encoder_jit_trace.pt
         - decoder_jit_trace.pt
         - joiner_jit_trace.pt

        Check ./jit_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--onnx",
        type=str2bool,
        default=False,
        help="""If True, --jit is ignored and it exports the model
        to onnx format. It will generate the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
            - joiner_encoder_proj.onnx
            - joiner_decoder_proj.onnx

        Refer to ./onnx_check.py and ./onnx_pretrained.py for how to use them.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
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


def export_encoder_model_jit_script(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.script()

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(encoder_model)
    script_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_script(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.script()

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(decoder_model)
    script_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_script(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.
    """
    script_model = torch.jit.script(joiner_model)
    script_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


def export_encoder_model_jit_trace(
    encoder_model: nn.Module,
    encoder_filename: str,
) -> None:
    """Export the given encoder model with torch.jit.trace()

    Note: The warmup argument is fixed to 1.

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported model.
    """
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

    traced_model = torch.jit.trace(encoder_model, (x, x_lens))
    traced_model.save(encoder_filename)
    logging.info(f"Saved to {encoder_filename}")


def export_decoder_model_jit_trace(
    decoder_model: nn.Module,
    decoder_filename: str,
) -> None:
    """Export the given decoder model with torch.jit.trace()

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The input decoder model
      decoder_filename:
        The filename to save the exported model.
    """
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
    need_pad = torch.tensor([False])

    traced_model = torch.jit.trace(decoder_model, (y, need_pad))
    traced_model.save(decoder_filename)
    logging.info(f"Saved to {decoder_filename}")


def export_joiner_model_jit_trace(
    joiner_model: nn.Module,
    joiner_filename: str,
) -> None:
    """Export the given joiner model with torch.jit.trace()

    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.

    Args:
      joiner_model:
        The input joiner model
      joiner_filename:
        The filename to save the exported model.

    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    traced_model = torch.jit.trace(joiner_model, (encoder_out, decoder_out))
    traced_model.save(joiner_filename)
    logging.info(f"Saved to {joiner_filename}")


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
    x = torch.zeros(1, 100, 80, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

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
    warmup = 1.0
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
    y = torch.zeros(10, decoder_model.context_size, dtype=torch.int64)
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

    projected_encoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(1, joiner_dim, dtype=torch.float32)

    project_input = False
    # Note: It uses torch.jit.trace() internally
    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out, project_input),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "projected_encoder_out",
            "projected_decoder_out",
            "project_input",
        ],
        output_names=["logit"],
        dynamic_axes={
            "projected_encoder_out": {0: "N"},
            "projected_decoder_out": {0: "N"},
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
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
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

    if params.onnx is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        opset_version = 11
        logging.info("Exporting to onnx format")
        encoder_filename = params.exp_dir / "encoder.onnx"
        export_encoder_model_onnx(
            model.encoder,
            encoder_filename,
            opset_version=opset_version,
        )

        decoder_filename = params.exp_dir / "decoder.onnx"
        export_decoder_model_onnx(
            model.decoder,
            decoder_filename,
            opset_version=opset_version,
        )

        joiner_filename = params.exp_dir / "joiner.onnx"
        export_joiner_model_onnx(
            model.joiner,
            joiner_filename,
            opset_version=opset_version,
        )
    elif params.jit is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.script()")
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

        # Also export encoder/decoder/joiner separately
        encoder_filename = params.exp_dir / "encoder_jit_script.pt"
        export_encoder_model_jit_script(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_script.pt"
        export_decoder_model_jit_script(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_script.pt"
        export_joiner_model_jit_script(model.joiner, joiner_filename)

    elif params.jit_trace is True:
        convert_scaled_to_non_scaled(model, inplace=True)
        logging.info("Using torch.jit.trace()")
        encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
        export_encoder_model_jit_trace(model.encoder, encoder_filename)

        decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
        export_decoder_model_jit_trace(model.decoder, decoder_filename)

        joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
        export_joiner_model_jit_trace(model.joiner, joiner_filename)
    else:
        logging.info("Not using torchscript")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
