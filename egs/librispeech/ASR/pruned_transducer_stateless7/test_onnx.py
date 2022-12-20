#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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


"""
This file is to test that models can be exported to onnx.
"""
import os

from icefall import is_module_available

if not is_module_available("onnxruntime"):
    raise ValueError("Please 'pip install onnxruntime' first.")

import onnxruntime as ort
import torch
from scaling_converter import convert_scaled_to_non_scaled
from zipformer import (
    Conv2dSubsampling,
    RelPositionalEncoding,
    Zipformer,
    ZipformerEncoder,
    ZipformerEncoderLayer,
)

ort.set_default_logger_severity(3)


def test_conv2d_subsampling():
    filename = "conv2d_subsampling.onnx"
    opset_version = 13
    N = 30
    T = 50
    num_features = 80
    d_model = 512
    x = torch.rand(N, T, num_features)

    encoder_embed = Conv2dSubsampling(num_features, d_model)
    encoder_embed.eval()
    encoder_embed = convert_scaled_to_non_scaled(encoder_embed, inplace=True)

    torch.onnx.export(
        encoder_embed,
        x,
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "y": {0: "N", 1: "T"},
        },
    )

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        filename,
        sess_options=options,
    )

    input_nodes = session.get_inputs()
    assert input_nodes[0].name == "x"
    assert input_nodes[0].shape == ["N", "T", num_features]

    inputs = {input_nodes[0].name: x.numpy()}

    onnx_y = session.run(["y"], inputs)[0]

    onnx_y = torch.from_numpy(onnx_y)
    torch_y = encoder_embed(x)
    assert torch.allclose(onnx_y, torch_y, atol=1e-05), (onnx_y - torch_y).abs().max()

    os.remove(filename)


def test_rel_pos():
    filename = "rel_pos.onnx"

    opset_version = 13
    N = 30
    T = 50
    num_features = 80
    d_model = 512
    x = torch.rand(N, T, num_features)

    encoder_pos = RelPositionalEncoding(d_model, dropout_rate=0.1)
    encoder_pos.eval()
    encoder_pos = convert_scaled_to_non_scaled(encoder_pos, inplace=True)

    x = x.permute(1, 0, 2)

    torch.onnx.export(
        encoder_pos,
        x,
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["pos_emb"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "pos_emb": {0: "N", 1: "T"},
        },
    )

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        filename,
        sess_options=options,
    )

    input_nodes = session.get_inputs()
    assert input_nodes[0].name == "x"
    assert input_nodes[0].shape == ["N", "T", num_features]

    inputs = {input_nodes[0].name: x.numpy()}
    onnx_pos_emb = session.run(["pos_emb"], inputs)
    onnx_pos_emb = torch.from_numpy(onnx_pos_emb[0])

    torch_pos_emb = encoder_pos(x)
    assert torch.allclose(onnx_pos_emb, torch_pos_emb, atol=1e-05), (
        (onnx_pos_emb - torch_pos_emb).abs().max()
    )
    print(onnx_pos_emb.abs().sum(), torch_pos_emb.abs().sum())

    os.remove(filename)


def test_zipformer_encoder_layer():
    filename = "zipformer_encoder_layer.onnx"
    opset_version = 13
    N = 30
    T = 50

    d_model = 384
    attention_dim = 192
    nhead = 8
    feedforward_dim = 1024
    dropout = 0.1
    cnn_module_kernel = 31
    pos_dim = 4

    x = torch.rand(N, T, d_model)

    encoder_pos = RelPositionalEncoding(d_model, dropout)
    encoder_pos.eval()
    encoder_pos = convert_scaled_to_non_scaled(encoder_pos, inplace=True)

    x = x.permute(1, 0, 2)
    pos_emb = encoder_pos(x)

    encoder_layer = ZipformerEncoderLayer(
        d_model,
        attention_dim,
        nhead,
        feedforward_dim,
        dropout,
        cnn_module_kernel,
        pos_dim,
    )
    encoder_layer.eval()
    encoder_layer = convert_scaled_to_non_scaled(encoder_layer, inplace=True)

    torch.onnx.export(
        encoder_layer,
        (x, pos_emb),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "pos_emb"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "T", 1: "N"},
            "pos_emb": {0: "N", 1: "T"},
            "y": {0: "T", 1: "N"},
        },
    )

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        filename,
        sess_options=options,
    )

    input_nodes = session.get_inputs()
    inputs = {
        input_nodes[0].name: x.numpy(),
        input_nodes[1].name: pos_emb.numpy(),
    }
    onnx_y = session.run(["y"], inputs)[0]
    onnx_y = torch.from_numpy(onnx_y)

    torch_y = encoder_layer(x, pos_emb)
    assert torch.allclose(onnx_y, torch_y, atol=1e-05), (onnx_y - torch_y).abs().max()

    print(onnx_y.abs().sum(), torch_y.abs().sum(), onnx_y.shape, torch_y.shape)

    os.remove(filename)


def test_zipformer_encoder():
    filename = "zipformer_encoder.onnx"

    opset_version = 13
    N = 3
    T = 15

    d_model = 512
    attention_dim = 192
    nhead = 8
    feedforward_dim = 1024
    dropout = 0.1
    cnn_module_kernel = 31
    pos_dim = 4
    num_encoder_layers = 12

    warmup_batches = 4000.0
    warmup_begin = warmup_batches / (num_encoder_layers + 1)
    warmup_end = warmup_batches / (num_encoder_layers + 1)

    x = torch.rand(N, T, d_model)

    encoder_layer = ZipformerEncoderLayer(
        d_model,
        attention_dim,
        nhead,
        feedforward_dim,
        dropout,
        cnn_module_kernel,
        pos_dim,
    )
    encoder = ZipformerEncoder(
        encoder_layer, num_encoder_layers, dropout, warmup_begin, warmup_end
    )
    encoder.eval()
    encoder = convert_scaled_to_non_scaled(encoder, inplace=True)

    # jit_model = torch.jit.trace(encoder, (pos_emb))

    torch_y = encoder(x)

    torch.onnx.export(
        encoder,
        (x),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "T", 1: "N"},
            "y": {0: "T", 1: "N"},
        },
    )

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        filename,
        sess_options=options,
    )

    input_nodes = session.get_inputs()
    inputs = {
        input_nodes[0].name: x.numpy(),
    }
    onnx_y = session.run(["y"], inputs)[0]
    onnx_y = torch.from_numpy(onnx_y)

    torch_y = encoder(x)
    assert torch.allclose(onnx_y, torch_y, atol=1e-05), (onnx_y - torch_y).abs().max()

    print(onnx_y.abs().sum(), torch_y.abs().sum(), onnx_y.shape, torch_y.shape)

    os.remove(filename)


def test_zipformer():
    filename = "zipformer.onnx"
    opset_version = 11
    N = 3
    T = 15
    num_features = 80
    x = torch.rand(N, T, num_features)
    x_lens = torch.full((N,), fill_value=T, dtype=torch.int64)

    zipformer = Zipformer(num_features=num_features)
    zipformer.eval()
    zipformer = convert_scaled_to_non_scaled(zipformer, inplace=True)

    # jit_model = torch.jit.trace(zipformer, (x, x_lens))
    torch.onnx.export(
        zipformer,
        (x, x_lens),
        filename,
        verbose=False,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["y", "y_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "y": {0: "N", 1: "T"},
            "y_lens": {0: "N"},
        },
    )
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1

    session = ort.InferenceSession(
        filename,
        sess_options=options,
    )

    input_nodes = session.get_inputs()
    inputs = {
        input_nodes[0].name: x.numpy(),
        input_nodes[1].name: x_lens.numpy(),
    }
    onnx_y, onnx_y_lens = session.run(["y", "y_lens"], inputs)
    onnx_y = torch.from_numpy(onnx_y)
    onnx_y_lens = torch.from_numpy(onnx_y_lens)

    torch_y, torch_y_lens = zipformer(x, x_lens)
    assert torch.allclose(onnx_y, torch_y, atol=1e-05), (onnx_y - torch_y).abs().max()

    assert torch.allclose(onnx_y_lens, torch_y_lens, atol=1e-05), (
        (onnx_y_lens - torch_y_lens).abs().max()
    )
    print(onnx_y.abs().sum(), torch_y.abs().sum(), onnx_y.shape, torch_y.shape)
    print(onnx_y_lens, torch_y_lens)

    os.remove(filename)


@torch.no_grad()
def main():
    test_conv2d_subsampling()
    test_rel_pos()
    test_zipformer_encoder_layer()
    test_zipformer_encoder()
    test_zipformer()


if __name__ == "__main__":
    torch.manual_seed(20221011)
    main()
