#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
import logging
from pathlib import Path
from typing import List

from rknn.api import RKNN
from test_rknn_on_cpu_simulator_ctc_streaming import (
    MetaData,
    get_meta_data,
    init_model,
    export_rknn,
)

logging.basicConfig(level=logging.WARNING)

g_platforms = [
    #  "rv1103",
    #  "rv1103b",
    #  "rv1106",
    #  "rk2118",
    "rk3562",
    "rk3566",
    "rk3568",
    "rk3576",
    "rk3588",
]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--target-platform",
        type=str,
        required=True,
        help=f"Supported values are: {','.join(g_platforms)}",
    )

    parser.add_argument(
        "--in-encoder",
        type=str,
        required=True,
        help="Path to the encoder onnx model",
    )

    parser.add_argument(
        "--in-decoder",
        type=str,
        required=True,
        help="Path to the decoder onnx model",
    )

    parser.add_argument(
        "--in-joiner",
        type=str,
        required=True,
        help="Path to the joiner onnx model",
    )

    parser.add_argument(
        "--out-encoder",
        type=str,
        required=True,
        help="Path to the encoder rknn model",
    )

    parser.add_argument(
        "--out-decoder",
        type=str,
        required=True,
        help="Path to the decoder rknn model",
    )

    parser.add_argument(
        "--out-joiner",
        type=str,
        required=True,
        help="Path to the joiner rknn model",
    )

    return parser


class RKNNModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
        target_platform: str,
    ):
        self.meta = get_meta_data(encoder)
        self.encoder = init_model(
            encoder,
            custom_string=self.meta.to_str(),
            target_platform=target_platform,
        )
        self.decoder = init_model(decoder, target_platform=target_platform)
        self.joiner = init_model(joiner, target_platform=target_platform)

    def export_rknn(self, encoder, decoder, joiner):
        export_rknn(self.encoder, encoder)
        export_rknn(self.decoder, decoder)
        export_rknn(self.joiner, joiner)

    def release(self):
        self.encoder.release()
        self.decoder.release()
        self.joiner.release()


def main():
    args = get_parser().parse_args()
    print(vars(args))

    model = RKNNModel(
        encoder=args.in_encoder,
        decoder=args.in_decoder,
        joiner=args.in_joiner,
        target_platform=args.target_platform,
    )
    print(model.meta)

    model.export_rknn(
        encoder=args.out_encoder,
        decoder=args.out_decoder,
        joiner=args.out_joiner,
    )

    model.release()


if __name__ == "__main__":
    main()
