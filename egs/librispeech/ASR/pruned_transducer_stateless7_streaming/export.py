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

./pruned_transducer_stateless7_streaming/export.py \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
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

./pruned_transducer_stateless7_streaming/export.py \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --epoch 20 \
  --avg 10

It will generate a file `pretrained.pt` in the given `exp_dir`. You can later
load it by `icefall.checkpoint.load_checkpoint()`.

To use the generated file with `pruned_transducer_stateless7_streaming/decode.py`,
you can do:

    cd /path/to/exp_dir
    ln -s pretrained.pt epoch-9999.pt

    cd /path/to/egs/librispeech/ASR
    ./pruned_transducer_stateless7_streaming/decode.py \
        --exp-dir ./pruned_transducer_stateless7_streaming/exp \
        --epoch 9999 \
        --avg 1 \
        --max-duration 600 \
        --decoding-method greedy_search \
        --bpe-model data/lang_bpe_500/bpe.model

Check ./pretrained.py for its usage.

Note: If you don't want to train a model from scratch, we have
provided one for you. You can get it at

https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29

with the following commands:

    sudo apt-get install git-lfs
    git lfs install
    git clone https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
    # You will find the pre-trained model in icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp

(3) Export to ONNX format with pretrained.pt

cd ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
ln -s pretrained.pt epoch-999.pt
./pruned_transducer_stateless7_streaming/export.py \
  --exp-dir ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --use-averaged-model False \
  --epoch 999 \
  --avg 1 \
  --fp16 \
  --onnx 1

It will generate the following files in the given `exp_dir`.
Check `onnx_check.py` for how to use them.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx
    - joiner_encoder_proj.onnx
    - joiner_decoder_proj.onnx

Check
https://github.com/k2-fsa/sherpa-onnx
for how to use the exported models outside of icefall.

(4) Export to ONNX format for triton server

cd ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp
ln -s pretrained.pt epoch-999.pt
./pruned_transducer_stateless7_streaming/export.py \
  --exp-dir ./icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp \
  --bpe-model data/lang_bpe_500/bpe.model \
  --use-averaged-model False \
  --epoch 999 \
  --avg 1 \
  --fp16 \
  --onnx-triton 1 \
  --onnx 1

It will generate the following files in the given `exp_dir`.
Check `onnx_check.py` for how to use them.

    - encoder.onnx
    - decoder.onnx
    - joiner.onnx

Check
https://github.com/k2-fsa/sherpa/tree/master/triton
for how to use the exported models outside of icefall.

"""


import argparse
import logging
from pathlib import Path

import onnxruntime
import sentencepiece as spm
import torch
import torch.nn as nn
from onnx_model_wrapper import OnnxStreamingEncoder, TritonOnnxDecoder, TritonOnnxJoiner
from scaling_converter import convert_scaled_to_non_scaled
from train import add_model_arguments, get_params, get_transducer_model
from zipformer import stack_states

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
        default="pruned_transducer_stateless7_streaming/exp",
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
        It will generate a file named cpu_jit.pt

        Check ./jit_pretrained.py for how to use it.
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
        "--onnx-triton",
        type=str2bool,
        default=False,
        help="""If True, --onnx would export model into the following files:

            - encoder.onnx
            - decoder.onnx
            - joiner.onnx
        These files would be used for https://github.com/k2-fsa/sherpa/tree/master/triton.
        """,
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="whether to export fp16 onnx model, default false",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


def test_acc(xlist, blist, rtol=1e-3, atol=1e-5, tolerate_small_mismatch=True):
    for a, b in zip(xlist, blist):
        try:
            torch.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
        except AssertionError as error:
            if tolerate_small_mismatch:
                print("small mismatch detected", error)
            else:
                return False
    return True


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
    batch_size = 17
    seq_len = 101
    torch.manual_seed(0)
    x = torch.rand(batch_size, seq_len, 80, dtype=torch.float32)
    x_lens = torch.tensor([seq_len - i for i in range(batch_size)], dtype=torch.int64)

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
    initial_states = [encoder_model.get_init_state() for _ in range(batch_size)]
    states = stack_states(initial_states)

    left_context_len = encoder_model.decode_chunk_size * encoder_model.num_left_chunks
    encoder_attention_dim = encoder_model.encoders[0].attention_dim

    len_cache = torch.cat(states[: encoder_model.num_encoders]).transpose(0, 1)  # B,15
    avg_cache = torch.cat(
        states[encoder_model.num_encoders : 2 * encoder_model.num_encoders]
    ).transpose(
        0, 1
    )  # [B,15,384]
    cnn_cache = torch.cat(states[5 * encoder_model.num_encoders :]).transpose(
        0, 1
    )  # [B,2*15,384,cnn_kernel-1]
    pad_tensors = [
        torch.nn.functional.pad(
            tensor,
            (
                0,
                encoder_attention_dim - tensor.shape[-1],
                0,
                0,
                0,
                left_context_len - tensor.shape[1],
                0,
                0,
            ),
        )
        for tensor in states[
            2 * encoder_model.num_encoders : 5 * encoder_model.num_encoders
        ]
    ]
    attn_cache = torch.cat(pad_tensors).transpose(0, 2)  # [B,64,15*3,192]

    encoder_model_wrapper = OnnxStreamingEncoder(encoder_model)

    torch.onnx.export(
        encoder_model_wrapper,
        (x, x_lens, len_cache, avg_cache, attn_cache, cnn_cache),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_lens",
            "len_cache",
            "avg_cache",
            "attn_cache",
            "cnn_cache",
        ],
        output_names=[
            "encoder_out",
            "encoder_out_lens",
            "new_len_cache",
            "new_avg_cache",
            "new_attn_cache",
            "new_cnn_cache",
        ],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "encoder_out": {0: "N", 1: "T"},
            "encoder_out_lens": {0: "N"},
            "len_cache": {0: "N"},
            "avg_cache": {0: "N"},
            "attn_cache": {0: "N"},
            "cnn_cache": {0: "N"},
            "new_len_cache": {0: "N"},
            "new_avg_cache": {0: "N"},
            "new_attn_cache": {0: "N"},
            "new_cnn_cache": {0: "N"},
        },
    )
    logging.info(f"Saved to {encoder_filename}")

    # Test onnx encoder with torch native encoder
    encoder_model.eval()
    (
        encoder_out_torch,
        encoder_out_lens_torch,
        new_states_torch,
    ) = encoder_model.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=states,
    )
    ort_session = onnxruntime.InferenceSession(
        str(encoder_filename), providers=["CPUExecutionProvider"]
    )
    ort_inputs = {
        "x": x.numpy(),
        "x_lens": x_lens.numpy(),
        "len_cache": len_cache.numpy(),
        "avg_cache": avg_cache.numpy(),
        "attn_cache": attn_cache.numpy(),
        "cnn_cache": cnn_cache.numpy(),
    }
    ort_outs = ort_session.run(None, ort_inputs)

    assert test_acc(
        [encoder_out_torch.numpy(), encoder_out_lens_torch.numpy()], ort_outs[:2]
    )
    logging.info(f"{encoder_filename} acc test succeeded.")


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


def export_decoder_model_onnx_triton(
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

    decoder_model = TritonOnnxDecoder(decoder_model)

    torch.onnx.export(
        decoder_model,
        (y,),
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
    The exported model has two inputs:
        - encoder_out: a tensor of shape (N, encoder_out_dim)
        - decoder_out: a tensor of shape (N, decoder_out_dim)
    and has one output:
        - joiner_out: a tensor of shape (N, vocab_size)
    Note: The argument project_input is fixed to True. A user should not
    project the encoder_out/decoder_out by himself/herself. The exported joiner
    will do that for the user.
    """
    encoder_out_dim = joiner_model.encoder_proj.weight.shape[1]
    decoder_out_dim = joiner_model.decoder_proj.weight.shape[1]
    encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
    decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

    joiner_model = TritonOnnxJoiner(joiner_model)
    # Note: It uses torch.jit.trace() internally
    torch.onnx.export(
        joiner_model,
        (encoder_out, decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    logging.info(f"Saved to {joiner_filename}")


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

    if params.onnx:
        convert_scaled_to_non_scaled(model, inplace=True)
        opset_version = 13
        logging.info("Exporting to onnx format")
        encoder_filename = params.exp_dir / "encoder.onnx"
        export_encoder_model_onnx(
            model.encoder,
            encoder_filename,
            opset_version=opset_version,
        )
        if not params.onnx_triton:
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
        else:
            decoder_filename = params.exp_dir / "decoder.onnx"
            export_decoder_model_onnx_triton(
                model.decoder,
                decoder_filename,
                opset_version=opset_version,
            )

            joiner_filename = params.exp_dir / "joiner.onnx"
            export_joiner_model_onnx_triton(
                model.joiner,
                joiner_filename,
                opset_version=opset_version,
            )

        if params.fp16:
            try:
                import onnxmltools
                from onnxmltools.utils.float16_converter import convert_float_to_float16
            except ImportError:
                print("Please install onnxmltools!")
                import sys

                sys.exit(1)

            def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
                onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
                onnx_fp16_model = convert_float_to_float16(onnx_fp32_model)
                onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)

            encoder_fp16_filename = params.exp_dir / "encoder_fp16.onnx"
            export_onnx_fp16(encoder_filename, encoder_fp16_filename)

            decoder_fp16_filename = params.exp_dir / "decoder_fp16.onnx"
            export_onnx_fp16(decoder_filename, decoder_fp16_filename)

            joiner_fp16_filename = params.exp_dir / "joiner_fp16.onnx"
            export_onnx_fp16(joiner_filename, joiner_fp16_filename)

            if not params.onnx_triton:
                encoder_proj_filename = str(joiner_filename).replace(
                    ".onnx", "_encoder_proj.onnx"
                )
                encoder_proj_fp16_filename = (
                    params.exp_dir / "joiner_encoder_proj_fp16.onnx"
                )
                export_onnx_fp16(encoder_proj_filename, encoder_proj_fp16_filename)

                decoder_proj_filename = str(joiner_filename).replace(
                    ".onnx", "_decoder_proj.onnx"
                )
                decoder_proj_fp16_filename = (
                    params.exp_dir / "joiner_decoder_proj_fp16.onnx"
                )
                export_onnx_fp16(decoder_proj_filename, decoder_proj_fp16_filename)

    elif params.jit:
        convert_scaled_to_non_scaled(model, inplace=True)
        # We won't use the forward() method of the model in C++, so just ignore
        # it here.
        # Otherwise, one of its arguments is a ragged tensor and is not
        # torch scriptabe.
        model.__class__.forward = torch.jit.ignore(model.__class__.forward)
        model.encoder.__class__.non_streaming_forward = model.encoder.__class__.forward
        model.encoder.__class__.non_streaming_forward = torch.jit.export(
            model.encoder.__class__.non_streaming_forward
        )
        model.encoder.__class__.forward = model.encoder.__class__.streaming_forward
        logging.info("Using torch.jit.script")
        model = torch.jit.script(model)
        filename = params.exp_dir / "cpu_jit.pt"
        model.save(str(filename))
        logging.info(f"Saved to {filename}")
    else:
        logging.info("Not using torchscript. Export model.state_dict()")
        # Save it using a format so that it can be loaded
        # by :func:`load_checkpoint`
        filename = params.exp_dir / "pretrained.pt"
        torch.save({"model": model.state_dict()}, str(filename))
        logging.info(f"Saved to {filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
