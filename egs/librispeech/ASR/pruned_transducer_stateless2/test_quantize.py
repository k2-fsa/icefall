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
    python ./pruned_transducer_stateless2/test_quantize.py
"""

import copy
import os

import torch
from quantize import (
    convert_scaled_linear,
    dynamic_quantize,
    scaled_linear_to_linear,
)
from scaling import ScaledLinear
from train import get_params, get_transducer_model


def get_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2

    params.dynamic_chunk_training = False
    params.short_chunk_size = 25
    params.num_left_chunks = 4
    params.causal_convolution = False

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


def test_convert_scaled_linear():
    for inplace in [False, True]:
        model = get_model()
        model.eval()

        orig_model = copy.deepcopy(model)

        converted_model = convert_scaled_linear(model, inplace=inplace)

        model = orig_model

        # test encoder
        N = 2
        T = 100
        vocab_size = model.decoder.vocab_size

        x = torch.randn(N, T, 80, dtype=torch.float32)
        x_lens = torch.full((N,), x.size(1))

        e1, e1_lens = model.encoder(x, x_lens)
        e2, e2_lens = converted_model.encoder(x, x_lens)

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


def test_dynamic_quantize_size_comparison():
    model = get_model()
    qmodel = dynamic_quantize(model)

    filename = "icefall-tmp-f32.pt"
    qfilename = "icefall-tmp-qin8.pt"
    torch.save(model, filename)
    torch.save(qmodel, qfilename)

    float_size = os.path.getsize(filename)
    int8_size = os.path.getsize(qfilename)
    print("float_size:", float_size)
    print("int8_size:", int8_size)
    print(f"ratio: {float_size}/{int8_size}: {float_size/int8_size:.3f}")

    os.remove(filename)
    os.remove(qfilename)


@torch.no_grad()
def main():
    test_scaled_linear_to_linear()
    test_convert_scaled_linear()
    test_dynamic_quantize_size_comparison()


if __name__ == "__main__":
    torch.manual_seed(20220725)
    main()
