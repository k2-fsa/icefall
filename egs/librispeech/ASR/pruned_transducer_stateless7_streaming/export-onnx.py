#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation (Author: Fangjun Kuang)

"""
This script exports a transducer model from PyTorch to ONNX.

We use the pre-trained model from
https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
as an example to show how to use this file.

1. Download the pre-trained model

cd egs/librispeech/ASR

repo_url=https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29
GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
repo=$(basename $repo_url)

pushd $repo
git lfs pull --include "data/lang_bpe_500/bpe.model"
git lfs pull --include "exp/pretrained.pt"
cd exp
ln -s pretrained.pt epoch-99.pt
popd

2. Export the model to ONNX

./pruned_transducer_stateless7_streaming/export-onnx.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --decode-chunk-len 32 \
  --exp-dir $repo/exp/

It will generate the following 3 files in $repo/exp

  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx

See ./onnx_pretrained.py for how to use the exported models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import k2
import onnx
import torch
import torch.nn as nn
from decoder import Decoder
from onnxruntime.quantization import QuantType, quantize_dynamic
from scaling_converter import convert_scaled_to_non_scaled
from torch import Tensor
from train import add_model_arguments, get_params, get_transducer_model
from zipformer import Zipformer

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import num_tokens, setup_logger, str2bool


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
        "--tokens",
        type=str,
        default="data/lang_bpe_500/tokens.txt",
        help="Path to the tokens.txt.",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    add_model_arguments(parser)

    return parser


class OnnxEncoder(nn.Module):
    """A wrapper for Zipformer and the encoder_proj from the joiner"""

    def __init__(self, encoder: Zipformer, encoder_proj: nn.Linear):
        """
        Args:
          encoder:
            A Zipformer encoder.
          encoder_proj:
            The projection layer for encoder from the joiner.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_proj = encoder_proj

    def forward(self, x: Tensor, states: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        """Please see the help information of Zipformer.streaming_forward"""
        N = x.size(0)
        T = x.size(1)
        x_lens = torch.tensor([T] * N, device=x.device)

        output, _, new_states = self.encoder.streaming_forward(
            x=x,
            x_lens=x_lens,
            states=states,
        )

        output = self.encoder_proj(output)
        # Now output is of shape (N, T, joiner_dim)

        return output, new_states


class OnnxDecoder(nn.Module):
    """A wrapper for Decoder and the decoder_proj from the joiner"""

    def __init__(self, decoder: Decoder, decoder_proj: nn.Linear):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, context_size).
        Returns
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        need_pad = False
        decoder_output = self.decoder(y, need_pad=need_pad)
        decoder_output = decoder_output.squeeze(1)
        output = self.decoder_proj(decoder_output)

        return output


class OnnxJoiner(nn.Module):
    """A wrapper for the joiner"""

    def __init__(self, output_linear: nn.Linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        logit = encoder_out + decoder_out
        logit = self.output_linear(torch.tanh(logit))
        return logit


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


def export_encoder_model_onnx(
    encoder_model: OnnxEncoder,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """
    Onnx model inputs:
      - 0: src
      - many state tensors (the exact number depending on the actual model)

    Onnx model outputs:
      - 0: output, its shape is (N, T, joiner_dim)
      - many state tensors (the exact number depending on the actual model)

    Args:
      encoder_model:
        The model to be exported
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """

    encoder_model.encoder.__class__.forward = (
        encoder_model.encoder.__class__.streaming_forward
    )

    decode_chunk_len = encoder_model.encoder.decode_chunk_size * 2
    pad_length = 7
    T = decode_chunk_len + pad_length
    logging.info(f"decode_chunk_len: {decode_chunk_len}")
    logging.info(f"pad_length: {pad_length}")
    logging.info(f"T: {T}")

    x = torch.rand(1, T, 80, dtype=torch.float32)

    init_state = encoder_model.encoder.get_init_state()

    num_encoders = encoder_model.encoder.num_encoders
    logging.info(f"num_encoders: {num_encoders}")
    logging.info(f"len(init_state): {len(init_state)}")

    inputs = {}
    input_names = ["x"]

    outputs = {}
    output_names = ["encoder_out"]

    def build_inputs_outputs(tensors, name, N):
        for i, s in enumerate(tensors):
            logging.info(f"{name}_{i}.shape: {s.shape}")
            inputs[f"{name}_{i}"] = {N: "N"}
            outputs[f"new_{name}_{i}"] = {N: "N"}
            input_names.append(f"{name}_{i}")
            output_names.append(f"new_{name}_{i}")

    num_encoder_layers = ",".join(map(str, encoder_model.encoder.num_encoder_layers))
    encoder_dims = ",".join(map(str, encoder_model.encoder.encoder_dims))
    attention_dims = ",".join(map(str, encoder_model.encoder.attention_dims))
    cnn_module_kernels = ",".join(map(str, encoder_model.encoder.cnn_module_kernels))
    ds = encoder_model.encoder.zipformer_downsampling_factors
    left_context_len = encoder_model.encoder.left_context_len
    left_context_len = [left_context_len // k for k in ds]
    left_context_len = ",".join(map(str, left_context_len))

    meta_data = {
        "model_type": "zipformer",
        "version": "1",
        "model_author": "k2-fsa",
        "decode_chunk_len": str(decode_chunk_len),  # 32
        "T": str(T),  # 39
        "num_encoder_layers": num_encoder_layers,
        "encoder_dims": encoder_dims,
        "attention_dims": attention_dims,
        "cnn_module_kernels": cnn_module_kernels,
        "left_context_len": left_context_len,
    }
    logging.info(f"meta_data: {meta_data}")

    # (num_encoder_layers, 1)
    cached_len = init_state[num_encoders * 0 : num_encoders * 1]

    # (num_encoder_layers, 1, encoder_dim)
    cached_avg = init_state[num_encoders * 1 : num_encoders * 2]

    # (num_encoder_layers, left_context_len, 1, attention_dim)
    cached_key = init_state[num_encoders * 2 : num_encoders * 3]

    # (num_encoder_layers, left_context_len, 1, attention_dim//2)
    cached_val = init_state[num_encoders * 3 : num_encoders * 4]

    # (num_encoder_layers, left_context_len, 1, attention_dim//2)
    cached_val2 = init_state[num_encoders * 4 : num_encoders * 5]

    # (num_encoder_layers, 1, encoder_dim, cnn_module_kernel-1)
    cached_conv1 = init_state[num_encoders * 5 : num_encoders * 6]

    # (num_encoder_layers, 1, encoder_dim, cnn_module_kernel-1)
    cached_conv2 = init_state[num_encoders * 6 : num_encoders * 7]

    build_inputs_outputs(cached_len, "cached_len", 1)
    build_inputs_outputs(cached_avg, "cached_avg", 1)
    build_inputs_outputs(cached_key, "cached_key", 2)
    build_inputs_outputs(cached_val, "cached_val", 2)
    build_inputs_outputs(cached_val2, "cached_val2", 2)
    build_inputs_outputs(cached_conv1, "cached_conv1", 1)
    build_inputs_outputs(cached_conv2, "cached_conv2", 1)

    logging.info(inputs)
    logging.info(outputs)
    logging.info(input_names)
    logging.info(output_names)

    torch.onnx.export(
        encoder_model,
        (x, init_state),
        encoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "x": {0: "N"},
            "encoder_out": {0: "N"},
            **inputs,
            **outputs,
        },
    )

    add_meta_data(filename=encoder_filename, meta_data=meta_data)


def export_decoder_model_onnx(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has one input:

        - y: a torch.int64 tensor of shape (N, context_size)

    and has one output:

        - decoder_out: a torch.float32 tensor of shape (N, joiner_dim)

    Note: The argument need_pad is fixed to False.

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    context_size = decoder_model.decoder.context_size
    vocab_size = decoder_model.decoder.vocab_size
    y = torch.zeros(10, context_size, dtype=torch.int64)
    decoder_model = torch.jit.script(decoder_model)
    torch.onnx.export(
        decoder_model,
        y,
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
    meta_data = {
        "context_size": str(context_size),
        "vocab_size": str(vocab_size),
    }
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.
    The exported joiner model has two inputs:

        - encoder_out: a tensor of shape (N, joiner_dim)
        - decoder_out: a tensor of shape (N, joiner_dim)

    and produces one output:

        - logit: a tensor of shape (N, vocab_size)
    """
    joiner_dim = joiner_model.output_linear.weight.shape[1]
    logging.info(f"joiner dim: {joiner_dim}")

    projected_encoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)
    projected_decoder_out = torch.rand(11, joiner_dim, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        (projected_encoder_out, projected_decoder_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=[
            "encoder_out",
            "decoder_out",
        ],
        output_names=["logit"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
            "logit": {0: "N"},
        },
    )
    meta_data = {
        "joiner_dim": str(joiner_dim),
    }
    add_meta_data(filename=joiner_filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    setup_logger(f"{params.exp_dir}/log-export/log-export-onnx")

    logging.info(f"device: {device}")

    # Load tokens.txt here
    token_table = k2.SymbolTable.from_file(params.tokens)

    # Load id of the <blk> token and the vocab size
    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]
    params.vocab_size = num_tokens(token_table) + 1  # +1 for <blk>

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
    encoder = OnnxEncoder(
        encoder=model.encoder,
        encoder_proj=model.joiner.encoder_proj,
    )

    decoder = OnnxDecoder(
        decoder=model.decoder,
        decoder_proj=model.joiner.decoder_proj,
    )

    joiner = OnnxJoiner(output_linear=model.joiner.output_linear)

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"
    if params.use_averaged_model:
        suffix += "-with-averaged-model"

    opset_version = 13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / f"encoder-{suffix}.onnx"
    export_encoder_model_onnx(
        encoder,
        encoder_filename,
        opset_version=opset_version,
    )
    logging.info(f"Exported encoder to {encoder_filename}")

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / f"decoder-{suffix}.onnx"
    export_decoder_model_onnx(
        decoder,
        decoder_filename,
        opset_version=opset_version,
    )
    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / f"joiner-{suffix}.onnx"
    export_joiner_model_onnx(
        joiner,
        joiner_filename,
        opset_version=opset_version,
    )
    logging.info(f"Exported joiner to {joiner_filename}")

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    logging.info("Generate int8 quantization models")

    encoder_filename_int8 = params.exp_dir / f"encoder-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=encoder_filename,
        model_output=encoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    decoder_filename_int8 = params.exp_dir / f"decoder-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=decoder_filename,
        model_output=decoder_filename_int8,
        op_types_to_quantize=["MatMul", "Gather"],
        weight_type=QuantType.QInt8,
    )

    joiner_filename_int8 = params.exp_dir / f"joiner-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=joiner_filename,
        model_output=joiner_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
