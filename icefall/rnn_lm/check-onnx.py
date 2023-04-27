#!/usr/bin/env python3
#
# Copyright 2023 Xiaomi Corporation

"""
Usage:

./check-onnx.py \
  --jit ./icefall-librispeech-rnn-lm/exp/cpu_jit.pt \
  --onnx ./icefall-librispeech-rnn-lm/exp/no-state-epoch-99-avg-1.onnx

Note: You can download pre-trained models from
https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm

"""

import argparse
import logging

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
        print(meta_data)

    def __call__(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        out = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0])


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.info(vars(args))

    torch_model = torch.jit.load(args.jit).cpu()
    onnx_model = OnnxModel(args.onnx)
    N = torch.arange(1, 5).tolist()

    for n in N:
        L = torch.randint(low=1, high=100, size=(1,)).item()
        x = torch.randint(
            low=1, high=onnx_model.vocab_size, size=(n, L), dtype=torch.int64
        )
        x_lens = torch.full((n,), fill_value=L, dtype=torch.int64)
        if n > 1:
            x_lens[0] = L // 2 + 1

        sos = torch.full((1,), fill_value=onnx_model.sos_id).expand(n, 1)
        sos_x = torch.cat([sos, x], dim=1)

        pad_col = torch.zeros((1,), dtype=x.dtype).expand(n, 1)
        x_eos = torch.cat([x, pad_col], dim=1)

        row_index = torch.arange(0, n, dtype=x.dtype)
        x_eos[row_index, x_lens] = onnx_model.eos_id

        torch_nll = torch_model(sos_x, x_eos, x_lens + 1).sum(dim=-1)
        onnx_nll = onnx_model(x, x_lens)
        # Note: For int8 models, the differences may be quite large,
        # e.g., within 0.9
        assert torch.allclose(torch_nll, onnx_nll), (
            torch_nll,
            onnx_nll,
        )
        print(n, L, torch_nll, onnx_nll)


if __name__ == "__main__":
    torch.manual_seed(20230420)
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
