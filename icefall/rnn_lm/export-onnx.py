#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation

import argparse
import logging
from pathlib import Path
from typing import Dict

import onnx
import torch
from model import RnnLmModel
from onnxruntime.quantization import QuantType, quantize_dynamic
from train import get_params

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import AttributeDict, str2bool


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


# A wrapper for RnnLm model to simpily the C++ calling code
# when exporting the model to ONNX.
#
# TODO(fangjun): The current wrapper works only for non-streaming ASR
# since we don't expose the LM state and it is used to score
# a complete sentence at once.
class RnnLmModelWrapper(torch.nn.Module):
    def __init__(self, model: RnnLmModel, sos_id: int, eos_id: int):
        super().__init__()
        self.model = model
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:
            A 2-D tensor of shape (N, L) with dtype torch.int64.
            It does not contain SOS or EOS. We will add SOS and EOS inside
            this function.
          x_lens:
            A 1-D tensor of shape (N,) with dtype torch.int64. It contains
            number of valid tokens in ``x`` before padding.
        Returns:
          Return a 1-D tensor of shape (N,) containing negative loglikelihood.
          Its dtype is torch.float32
        """
        N = x.size(0)

        sos_tensor = torch.full((1,), fill_value=self.sos_id, dtype=x.dtype).expand(
            N, 1
        )
        sos_x = torch.cat([sos_tensor, x], dim=1)

        pad_col = torch.zeros((1,), dtype=x.dtype).expand(N, 1)
        x_eos = torch.cat([x, pad_col], dim=1)

        row_index = torch.arange(0, N, dtype=x.dtype)
        x_eos[row_index, x_lens] = self.eos_id

        # use x_lens + 1 here since we prepended x with sos
        return (
            self.model(x=sos_x, y=x_eos, lengths=x_lens + 1)
            .to(torch.float32)
            .sum(dim=1)
        )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=29,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
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
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the model",
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of RNN layers the model",
    )

    parser.add_argument(
        "--tie-weights",
        type=str2bool,
        default=True,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    return parser


def export_without_state(
    model: RnnLmModel,
    filename: str,
    params: AttributeDict,
    opset_version: int,
):
    model_wrapper = RnnLmModelWrapper(
        model,
        sos_id=params.sos_id,
        eos_id=params.eos_id,
    )

    N = 1
    L = 20
    x = torch.randint(low=1, high=params.vocab_size, size=(N, L), dtype=torch.int64)
    x_lens = torch.full((N,), fill_value=L, dtype=torch.int64)

    # Note(fangjun): The following warnings can be ignored.
    # We can use ./check-onnx.py to validate the exported model with batch_size > 1
    """
    torch/onnx/symbolic_opset9.py:2119: UserWarning: Exporting a model to ONNX
    with a batch_size other than 1, with a variable length with LSTM can cause
    an error when running the ONNX model with a different batch size. Make sure
    to save the model with a batch size of 1, or define the initial states
    (h0/c0) as inputs of the model. warnings.warn("Exporting a model to ONNX
    with a batch_size other than 1, " +
    """

    torch.onnx.export(
        model_wrapper,
        (x, x_lens),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["nll"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lens": {0: "N"},
            "nll": {0: "N"},
        },
    )

    meta_data = {
        "model_type": "rnnlm",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "rnnlm without state",
        "sos_id": str(params.sos_id),
        "eos_id": str(params.eos_id),
        "vocab_size": str(params.vocab_size),
        "url": "https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm",
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=filename, meta_data=meta_data)


def export_with_state(
    model: RnnLmModel,
    filename: str,
    params: AttributeDict,
    opset_version: int,
):
    N = 1
    L = 20
    num_layers = model.rnn.num_layers
    hidden_size = model.rnn.hidden_size
    embedding_dim = model.embedding_dim

    x = torch.randint(low=1, high=params.vocab_size, size=(N, L), dtype=torch.int64)
    h0 = torch.zeros(num_layers, N, hidden_size)
    c0 = torch.zeros(num_layers, N, hidden_size)

    # Note(fangjun): The following warnings can be ignored.
    # We can use ./check-onnx.py to validate the exported model with batch_size > 1
    """
    torch/onnx/symbolic_opset9.py:2119: UserWarning: Exporting a model to ONNX
    with a batch_size other than 1, with a variable length with LSTM can cause
    an error when running the ONNX model with a different batch size. Make sure
    to save the model with a batch size of 1, or define the initial states
    (h0/c0) as inputs of the model. warnings.warn("Exporting a model to ONNX
    with a batch_size other than 1, " +
    """

    torch.onnx.export(
        model,
        (x, h0, c0),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "h0", "c0"],
        output_names=["log_softmax", "next_h0", "next_c0"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "h0": {1: "N"},
            "c0": {1: "N"},
            "log_softmax": {0: "N"},
            "next_h0": {1: "N"},
            "next_c0": {1: "N"},
        },
    )

    meta_data = {
        "model_type": "rnnlm",
        "version": "1",
        "model_author": "k2-fsa",
        "comment": "rnnlm state",
        "sos_id": str(params.sos_id),
        "eos_id": str(params.eos_id),
        "vocab_size": str(params.vocab_size),
        "num_layers": str(num_layers),
        "hidden_size": str(hidden_size),
        "embedding_dim": str(embedding_dim),
        "url": "https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm",
    }
    logging.info(f"meta_data: {meta_data}")

    add_meta_data(filename=filename, meta_data=meta_data)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    logging.info(params)

    device = torch.device("cpu")
    logging.info(f"device: {device}")

    model = RnnLmModel(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        tie_weights=params.tie_weights,
    )

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
            if i >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to("cpu")
    model.eval()

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"

    opset_version = 13

    logging.info("Exporting model without state")
    filename = params.exp_dir / f"no-state-{suffix}.onnx"
    export_without_state(
        model=model,
        filename=filename,
        params=params,
        opset_version=opset_version,
    )

    filename_int8 = params.exp_dir / f"no-state-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QInt8,
    )

    # now for streaming export
    saved_forward = model.__class__.forward
    model.__class__.forward = model.__class__.score_token_onnx
    streaming_filename = params.exp_dir / f"with-state-{suffix}.onnx"
    export_with_state(
        model=model,
        filename=streaming_filename,
        params=params,
        opset_version=opset_version,
    )
    model.__class__.forward = saved_forward

    streaming_filename_int8 = params.exp_dir / f"with-state-{suffix}.int8.onnx"
    quantize_dynamic(
        model_input=streaming_filename,
        model_output=streaming_filename_int8,
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
