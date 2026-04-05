#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

import argparse
import logging
from pathlib import Path
from typing import List

from rknn.api import RKNN

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


def export_rknn(rknn, filename):
    ret = rknn.export_rknn(filename)
    if ret != 0:
        exit("Export rknn model to {filename} failed!")


def init_model(filename: str, target_platform: str, custom_string=None):
    rknn = RKNN(verbose=False)

    rknn.config(target_platform=target_platform, custom_string=custom_string)
    if not Path(filename).is_file():
        exit(f"{filename} does not exist")

    ret = rknn.load_onnx(model=filename)
    if ret != 0:
        exit(f"Load model {filename} failed!")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        exit("Build model {filename} failed!")

    return rknn


class MetaData:
    def __init__(
        self,
        model_type: str,
        attention_dims: List[int],
        encoder_dims: List[int],
        T: int,
        left_context_len: List[int],
        decode_chunk_len: int,
        cnn_module_kernels: List[int],
        num_encoder_layers: List[int],
        context_size: int,
    ):
        self.model_type = model_type
        self.attention_dims = attention_dims
        self.encoder_dims = encoder_dims
        self.T = T
        self.left_context_len = left_context_len
        self.decode_chunk_len = decode_chunk_len
        self.cnn_module_kernels = cnn_module_kernels
        self.num_encoder_layers = num_encoder_layers
        self.context_size = context_size

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        def to_s(ll):
            return ",".join(list(map(str, ll)))

        s = f"model_type={self.model_type}"
        s += ";attention_dims=" + to_s(self.attention_dims)
        s += ";encoder_dims=" + to_s(self.encoder_dims)
        s += ";T=" + str(self.T)
        s += ";left_context_len=" + to_s(self.left_context_len)
        s += ";decode_chunk_len=" + str(self.decode_chunk_len)
        s += ";cnn_module_kernels=" + to_s(self.cnn_module_kernels)
        s += ";num_encoder_layers=" + to_s(self.num_encoder_layers)
        s += ";context_size=" + str(self.context_size)

        assert len(s) < 1024, (s, len(s))

        return s


def get_meta_data(encoder: str, decoder: str):
    import onnxruntime

    session_opts = onnxruntime.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    m_encoder = onnxruntime.InferenceSession(
        encoder,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    m_decoder = onnxruntime.InferenceSession(
        decoder,
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )

    encoder_meta = m_encoder.get_modelmeta().custom_metadata_map
    print(encoder_meta)

    # {'attention_dims': '192,192,192,192,192', 'version': '1',
    # 'model_type': 'zipformer', 'encoder_dims': '256,256,256,256,256',
    # 'model_author': 'k2-fsa', 'T': '103',
    # 'left_context_len': '192,96,48,24,96',
    # 'decode_chunk_len': '96',
    # 'cnn_module_kernels': '31,31,31,31,31',
    # 'num_encoder_layers': '2,2,2,2,2'}

    def to_int_list(s):
        return list(map(int, s.split(",")))

    decoder_meta = m_decoder.get_modelmeta().custom_metadata_map
    print(decoder_meta)

    model_type = encoder_meta["model_type"]
    attention_dims = to_int_list(encoder_meta["attention_dims"])
    encoder_dims = to_int_list(encoder_meta["encoder_dims"])
    T = int(encoder_meta["T"])
    left_context_len = to_int_list(encoder_meta["left_context_len"])
    decode_chunk_len = int(encoder_meta["decode_chunk_len"])
    cnn_module_kernels = to_int_list(encoder_meta["cnn_module_kernels"])
    num_encoder_layers = to_int_list(encoder_meta["num_encoder_layers"])
    context_size = int(decoder_meta["context_size"])

    return MetaData(
        model_type=model_type,
        attention_dims=attention_dims,
        encoder_dims=encoder_dims,
        T=T,
        left_context_len=left_context_len,
        decode_chunk_len=decode_chunk_len,
        cnn_module_kernels=cnn_module_kernels,
        num_encoder_layers=num_encoder_layers,
        context_size=context_size,
    )


class RKNNModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
        target_platform: str,
    ):
        self.meta = get_meta_data(encoder, decoder)
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
