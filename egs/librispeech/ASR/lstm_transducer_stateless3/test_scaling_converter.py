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
To run this file, do:

    cd icefall/egs/librispeech/ASR
    python ./lstm_transducer_stateless/test_scaling_converter.py
"""

import copy

import torch
from scaling import (
    ScaledConv1d,
    ScaledConv2d,
    ScaledEmbedding,
    ScaledLinear,
    ScaledLSTM,
)
from scaling_converter import (
    convert_scaled_to_non_scaled,
    scaled_conv1d_to_conv1d,
    scaled_conv2d_to_conv2d,
    scaled_embedding_to_embedding,
    scaled_linear_to_linear,
    scaled_lstm_to_lstm,
)
from train import get_params, get_transducer_model


def get_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2
    params.encoder_dim = 512
    params.rnn_hidden_size = 1024
    params.num_encoder_layers = 12
    params.aux_layer_period = -1

    model = get_transducer_model(params)
    return model


def test_scaled_linear_to_linear():
    N = 5
    in_features = 10
    out_features = 20
    for bias in [True, False]:
        scaled_linear = ScaledLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        linear = scaled_linear_to_linear(scaled_linear)
        x = torch.rand(N, in_features)

        y1 = scaled_linear(x)
        y2 = linear(x)
        assert torch.allclose(y1, y2)

        jit_scaled_linear = torch.jit.script(scaled_linear)
        jit_linear = torch.jit.script(linear)

        y3 = jit_scaled_linear(x)
        y4 = jit_linear(x)

        assert torch.allclose(y3, y4)
        assert torch.allclose(y1, y4)


def test_scaled_conv1d_to_conv1d():
    in_channels = 3
    for bias in [True, False]:
        scaled_conv1d = ScaledConv1d(
            in_channels,
            6,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        conv1d = scaled_conv1d_to_conv1d(scaled_conv1d)

        x = torch.rand(20, in_channels, 10)
        y1 = scaled_conv1d(x)
        y2 = conv1d(x)
        assert torch.allclose(y1, y2)

        jit_scaled_conv1d = torch.jit.script(scaled_conv1d)
        jit_conv1d = torch.jit.script(conv1d)

        y3 = jit_scaled_conv1d(x)
        y4 = jit_conv1d(x)

        assert torch.allclose(y3, y4)
        assert torch.allclose(y1, y4)


def test_scaled_conv2d_to_conv2d():
    in_channels = 1
    for bias in [True, False]:
        scaled_conv2d = ScaledConv2d(
            in_channels=in_channels,
            out_channels=3,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        conv2d = scaled_conv2d_to_conv2d(scaled_conv2d)

        x = torch.rand(20, in_channels, 10, 20)
        y1 = scaled_conv2d(x)
        y2 = conv2d(x)
        assert torch.allclose(y1, y2)

        jit_scaled_conv2d = torch.jit.script(scaled_conv2d)
        jit_conv2d = torch.jit.script(conv2d)

        y3 = jit_scaled_conv2d(x)
        y4 = jit_conv2d(x)

        assert torch.allclose(y3, y4)
        assert torch.allclose(y1, y4)


def test_scaled_embedding_to_embedding():
    scaled_embedding = ScaledEmbedding(
        num_embeddings=500,
        embedding_dim=10,
        padding_idx=0,
    )
    embedding = scaled_embedding_to_embedding(scaled_embedding)

    for s in [10, 100, 300, 500, 800, 1000]:
        x = torch.randint(low=0, high=500, size=(s,))
        scaled_y = scaled_embedding(x)
        y = embedding(x)
        assert torch.equal(scaled_y, y)


def test_scaled_lstm_to_lstm():
    input_size = 512
    batch_size = 20
    for bias in [True, False]:
        for hidden_size in [512, 1024]:
            scaled_lstm = ScaledLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                bias=bias,
                proj_size=0 if hidden_size == input_size else input_size,
            )

            lstm = scaled_lstm_to_lstm(scaled_lstm)

            x = torch.rand(200, batch_size, input_size)
            h0 = torch.randn(1, batch_size, input_size)
            c0 = torch.randn(1, batch_size, hidden_size)

            y1, (h1, c1) = scaled_lstm(x, (h0, c0))
            y2, (h2, c2) = lstm(x, (h0, c0))
            assert torch.allclose(y1, y2)
            assert torch.allclose(h1, h2)
            assert torch.allclose(c1, c2)

            jit_scaled_lstm = torch.jit.trace(lstm, (x, (h0, c0)))
            y3, (h3, c3) = jit_scaled_lstm(x, (h0, c0))
            assert torch.allclose(y1, y3)
            assert torch.allclose(h1, h3)
            assert torch.allclose(c1, c3)


def test_convert_scaled_to_non_scaled():
    for inplace in [False, True]:
        model = get_model()
        model.eval()

        orig_model = copy.deepcopy(model)

        converted_model = convert_scaled_to_non_scaled(model, inplace=inplace)

        model = orig_model

        # test encoder
        N = 2
        T = 100
        vocab_size = model.decoder.vocab_size

        x = torch.randn(N, T, 80, dtype=torch.float32)
        x_lens = torch.full((N,), x.size(1))

        e1, e1_lens, _ = model.encoder(x, x_lens)
        e2, e2_lens, _ = converted_model.encoder(x, x_lens)

        assert torch.all(torch.eq(e1_lens, e2_lens))
        assert torch.allclose(e1, e2), (e1 - e2).abs().max()

        # test decoder
        U = 50
        y = torch.randint(low=1, high=vocab_size - 1, size=(N, U))

        d1 = model.decoder(y)
        d2 = model.decoder(y)

        assert torch.allclose(d1, d2)

        # test simple projection
        lm1 = model.simple_lm_proj(d1)
        am1 = model.simple_am_proj(e1)

        lm2 = converted_model.simple_lm_proj(d2)
        am2 = converted_model.simple_am_proj(e2)

        assert torch.allclose(lm1, lm2)
        assert torch.allclose(am1, am2)

        # test joiner
        e = torch.rand(2, 3, 4, 512)
        d = torch.rand(2, 3, 4, 512)

        j1 = model.joiner(e, d)
        j2 = converted_model.joiner(e, d)
        assert torch.allclose(j1, j2)


@torch.no_grad()
def main():
    test_scaled_linear_to_linear()
    test_scaled_conv1d_to_conv1d()
    test_scaled_conv2d_to_conv2d()
    test_scaled_embedding_to_embedding()
    test_scaled_lstm_to_lstm()
    test_convert_scaled_to_non_scaled()


if __name__ == "__main__":
    torch.manual_seed(20220730)
    main()
