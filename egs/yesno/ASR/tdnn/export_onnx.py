#!/usr/bin/env python3

"""
This file is for exporting trained models to onnx.

Usage:

    ./tdnn/export_onnx.py \
      --epoch 14 \
      --avg 2

The above command generates the following two files:
  - ./exp/model-epoch-14-avg-2.onnx
  - ./exp/model-epoch-14-avg-2.int8.onnx

See ./tdnn/onnx_pretrained.py for how to use them.
"""

import argparse
import logging
from typing import Dict

import onnx
import torch
from model import Tdnn
from onnxruntime.quantization import QuantType, quantize_dynamic
from train import get_params

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.lexicon import Lexicon


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=14,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=2,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    return parser


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
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    args = get_parser().parse_args()

    params = get_params()
    params.update(vars(args))

    logging.info(params)

    lexicon = Lexicon(params.lang_dir)
    max_token_id = max(lexicon.tokens)

    model = Tdnn(
        num_features=params.feature_dim,
        num_classes=max_token_id + 1,  # +1 for the blank symbol
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
    model.eval()

    N = 1
    T = 100
    C = params.feature_dim
    x = torch.rand(N, T, C)

    opset_version = 13
    onnx_filename = f"{params.exp_dir}/model-epoch-{params.epoch}-avg-{params.avg}.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["log_prob"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "log_prob": {0: "N", 1: "T"},
        },
    )

    logging.info(f"Saved to {onnx_filename}")
    meta_data = {
        "model_type": "tdnn",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "non-streaming tdnn for the yesno recipe",
        "vocab_size": max_token_id + 1,
    }

    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=onnx_filename, meta_data=meta_data)

    logging.info("Generate int8 quantization models")
    onnx_filename_int8 = (
        f"{params.exp_dir}/model-epoch-{params.epoch}-avg-{params.avg}.int8.onnx"
    )

    quantize_dynamic(
        model_input=onnx_filename,
        model_output=onnx_filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )
    logging.info(f"Saved to {onnx_filename_int8}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
