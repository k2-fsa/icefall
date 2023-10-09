#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation

"""
Usage:

./check-onnx-streaming.py \
  --jit ./icefall-librispeech-rnn-lm/exp/cpu_jit.pt \
  --onnx ./icefall-librispeech-rnn-lm/exp/with-state-epoch-99-avg-1.onnx

Note: You can download pre-trained models from
https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm

"""

import argparse
import logging
from typing import Tuple

import onnxruntime as ort
import torch


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--jit",
        required=True,
        type=str,
        help="Path to the torchscript model",
    )

    parser.add_argument(
        "--onnx",
        required=True,
        type=str,
        help="Path to the onnx model",
    )

    return parser


class OnnxModel:
    def __init__(self, filename: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.model = ort.InferenceSession(
            filename,
            sess_options=session_opts,
        )

        meta_data = self.model.get_modelmeta().custom_metadata_map
        self.sos_id = int(meta_data["sos_id"])
        self.eos_id = int(meta_data["eos_id"])
        self.vocab_size = int(meta_data["vocab_size"])
        self.num_layers = int(meta_data["num_layers"])
        self.hidden_size = int(meta_data["hidden_size"])
        print(meta_data)

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: y.numpy(),
                self.model.get_inputs()[2].name: h0.numpy(),
                self.model.get_inputs()[3].name: c0.numpy(),
            },
        )
        return (
            torch.from_numpy(out[0]),
            torch.from_numpy(out[1]),
            torch.from_numpy(out[2]),
        )


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    torch_model = torch.jit.load(args.jit).cpu()
    onnx_model = OnnxModel(args.onnx)
    N = torch.arange(1, 5).tolist()

    num_layers = onnx_model.num_layers
    hidden_size = onnx_model.hidden_size

    for n in N:
        L = torch.randint(low=1, high=100, size=(1,)).item()
        x = torch.randint(
            low=1, high=onnx_model.vocab_size, size=(n, L), dtype=torch.int64
        )
        h0 = torch.rand(num_layers, n, hidden_size)
        c0 = torch.rand(num_layers, n, hidden_size)

        torch_log_prob, torch_h0, torch_c0 = torch_model.score_token_onnx(x, h0, c0)
        onnx_log_prob, onnx_h0, onnx_c0 = onnx_model(x, h0, c0)

        for torch_v, onnx_v in zip(
            (torch_log_prob, torch_h0, torch_c0), (onnx_log_prob, onnx_h0, onnx_c0)
        ):
            assert torch.allclose(torch_v, onnx_v, atol=1e-5), (
                torch_v.shape,
                onnx_v.shape,
                (torch_v - onnx_v).abs().max(),
            )
            print(n, L, torch_v.sum(), onnx_v.sum())


if __name__ == "__main__":
    torch.manual_seed(20230423)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
