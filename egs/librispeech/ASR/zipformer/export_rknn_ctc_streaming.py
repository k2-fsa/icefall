#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
import logging
from pathlib import Path
from typing import List

from rknn.api import RKNN
from test_rknn_on_cpu_simulator_ctc_streaming import RKNNModel

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
        "--in-model",
        type=str,
        required=True,
        help="Path to the onnx model",
    )

    parser.add_argument(
        "--out-model",
        type=str,
        required=True,
        help="Path to the rknn model",
    )

    return parser


def main():
    args = get_parser().parse_args()
    print(vars(args))

    model = RKNNModel(
        model=args.in_model,
        target_platform=args.target_platform,
    )
    print(model.meta)

    model.export_rknn(
        model=args.out_model,
    )

    model.release()


if __name__ == "__main__":
    main()
