#!/usr/bin/env python3
#
# Copyright 2025 Xiaomi Corporation (Author: Fangjun Kuang)

import logging
from pathlib import Path
from typing import Dict, Tuple

import onnx
import torch
from decode import get_params, get_parser, get_transducer_model
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transducer.conformer import Conformer

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class OnnxEncoder(nn.Module):
    """A wrapper for Conformer"""

    def __init__(self, encoder: Conformer):
        """
        Args:
          encoder:
            A Conformer encoder.
        """
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Please see the help information of Conformer.forward

        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 1-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a tuple containing:
            - encoder_out, A 3-D tensor of shape (N, T', joiner_dim)
            - encoder_out_lens, A 1-D tensor of shape (N,)
        """
        encoder_out, encoder_out_lens = self.encoder(x, x_lens)
        # Now encoder_out is of shape (N, T, joiner_dim)

        return encoder_out, encoder_out_lens


def export_encoder_model_onnx(
    encoder_model: OnnxEncoder,
    encoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T', C)
        - encoder_out_lens, a tensor of shape (N,)

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    x = torch.zeros(1, 100, 23, dtype=torch.float32)
    x_lens = torch.tensor([100], dtype=torch.int64)

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

    meta_data = {
        "model_type": "conformer",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "SLU_transducer",
        "note": "The decoder is an LSTM with states",
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=encoder_filename, meta_data=meta_data)


def export_decoder_model_onnx(
    decoder_model: nn.Module,
    decoder_filename: str,
    opset_version: int = 11,
) -> None:
    """Export the decoder model to ONNX format.

    The exported model has 3 inputs:

        - y: a torch.int64 tensor of shape (N, 1)
        - h: a float32 tensor of shape (num_layers, N, hidden_dim)
        - c: a float32 tensor of shape (num_layers, N, hidden_dim)

    and has 3 outputs:

        - decoder_out: a torch.float32 tensor of shape (N, 1, C)

    Args:
      decoder_model:
        The decoder model to be exported.
      decoder_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    y = torch.zeros(1, 1, dtype=torch.int64)
    _, (h, c) = decoder_model(y)

    torch.onnx.export(
        decoder_model,
        (y, (h, c)),
        decoder_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["y", "h", "c"],
        output_names=["decoder_out", "next_h", "next_c"],
        dynamic_axes={
            "y": {0: "N"},
            "h": {1: "N"},
            "c": {1: "N"},
        },
    )

    meta_data = {
        "num_layers": h.shape[0],
        "hidden_dim": h.shape[2],
    }
    print("decoder meta_data", meta_data)
    add_meta_data(filename=decoder_filename, meta_data=meta_data)


def export_joiner_model_onnx(
    joiner_model: nn.Module,
    joiner_filename: str,
    hidden_dim: int,
    vocab_size: int,
    opset_version: int = 11,
) -> None:
    """Export the joiner model to ONNX format.

    The exported model has 2 inputs:

        - encoder_out: a float32 tensor of shape (N, 1, enc_dim)
        - decoder_out: a float32 tensor of shape (N, 1, dec_dim)

    and has 1 output:

        - logits: a torch.float32 tensor of shape (N, 1, vocab_size)

    Args:
      joiner_model:
        The decoder model to be exported.
      joiner_filename:
        Filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
    enc_out = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)
    dec_out = torch.zeros(1, 1, hidden_dim, dtype=torch.float32)

    torch.onnx.export(
        joiner_model,
        (enc_out, dec_out),
        joiner_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["encoder_out", "decoder_out"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_out": {0: "N"},
            "decoder_out": {0: "N"},
        },
    )

    meta_data = {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
    }
    print("joiner meta_data", meta_data)
    add_meta_data(filename=joiner_filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))
    params["env_info"] = get_env_info()

    device = torch.device("cpu")
    logging.info(f"device: {device}")
    model = get_transducer_model(params)

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

    model.to(device)
    model.eval()
    model.device = device

    encoder_num_param = sum([p.numel() for p in model.encoder.parameters()])
    decoder_num_param = sum([p.numel() for p in model.decoder.parameters()])
    joiner_num_param = sum([p.numel() for p in model.joiner.parameters()])
    total_num_param = encoder_num_param + decoder_num_param + joiner_num_param
    logging.info(f"encoder parameters: {encoder_num_param}")
    logging.info(f"decoder parameters: {decoder_num_param}")
    logging.info(f"joiner parameters: {joiner_num_param}")
    logging.info(f"total parameters: {total_num_param}")

    encoder = OnnxEncoder(
        encoder=model.encoder,
    )

    opset_version = 13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / "encoder.onnx"
    #  export_encoder_model_onnx(
    #      encoder,
    #      encoder_filename,
    #      opset_version=opset_version,
    #  )
    logging.info(f"Exported encoder to {encoder_filename}")

    logging.info("Exporting decoder")
    decoder_filename = params.exp_dir / "decoder.onnx"
    #  export_decoder_model_onnx(
    #      model.decoder,
    #      decoder_filename,
    #      opset_version=opset_version,
    #  )
    logging.info(f"Exported decoder to {decoder_filename}")

    logging.info("Exporting joiner")
    joiner_filename = params.exp_dir / "joiner.onnx"
    export_joiner_model_onnx(
        model.joiner,
        joiner_filename,
        opset_version=opset_version,
        hidden_dim=params.hidden_dim,
        vocab_size=params.vocab_size,
    )
    logging.info(f"Exported decoder to {joiner_filename}")

    # Generate int8 quantization models
    # See https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection

    logging.info("Generate int8 quantization models")

    encoder_filename_int8 = params.exp_dir / "encoder.int8.onnx"
    quantize_dynamic(
        model_input=encoder_filename,
        model_output=encoder_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    decoder_filename_int8 = params.exp_dir / "decoder.int8.onnx"
    quantize_dynamic(
        model_input=decoder_filename,
        model_output=decoder_filename_int8,
        op_types_to_quantize=["MatMul", "Gather"],
        weight_type=QuantType.QInt8,
    )

    joiner_filename_int8 = params.exp_dir / "joiner.int8.onnx"
    quantize_dynamic(
        model_input=joiner_filename,
        model_output=joiner_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
