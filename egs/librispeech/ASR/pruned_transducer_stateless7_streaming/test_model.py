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
    python ./pruned_transducer_stateless7_streaming/test_model.py
"""

import torch
from scaling_converter import convert_scaled_to_non_scaled
from train import get_params, get_transducer_model


def test_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,4,3,2,4"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    params.decoder_dim = 512
    params.joiner_dim = 512
    params.num_left_chunks = 4
    params.short_chunk_size = 50
    params.decode_chunk_len = 32
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    # Test jit script
    convert_scaled_to_non_scaled(model, inplace=True)
    # We won't use the forward() method of the model in C++, so just ignore
    # it here.
    # Otherwise, one of its arguments is a ragged tensor and is not
    # torch scriptabe.
    model.__class__.forward = torch.jit.ignore(model.__class__.forward)
    print("Using torch.jit.script")
    model = torch.jit.script(model)


def test_model_small():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,2,2,2,2"
    params.feedforward_dims = "256,256,512,512,256"
    params.nhead = "4,4,4,4,4"
    params.encoder_dims = "128,128,128,128,128"
    params.attention_dims = "96,96,96,96,96"
    params.encoder_unmasked_dims = "96,96,96,96,96"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    params.decoder_dim = 320
    params.joiner_dim = 320
    params.num_left_chunks = 4
    params.short_chunk_size = 50
    params.decode_chunk_len = 32
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    import pdb

    pdb.set_trace()

    # Test jit script
    convert_scaled_to_non_scaled(model, inplace=True)
    # We won't use the forward() method of the model in C++, so just ignore
    # it here.
    # Otherwise, one of its arguments is a ragged tensor and is not
    # torch scriptabe.
    model.__class__.forward = torch.jit.ignore(model.__class__.forward)
    print("Using torch.jit.script")
    model = torch.jit.script(model)


def test_model_jit_trace():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.num_encoder_layers = "2,4,3,2,4"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    params.decoder_dim = 512
    params.joiner_dim = 512
    params.num_left_chunks = 4
    params.short_chunk_size = 50
    params.decode_chunk_len = 32
    model = get_transducer_model(params)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    convert_scaled_to_non_scaled(model, inplace=True)

    # Test encoder
    def _test_encoder():
        encoder = model.encoder
        assert encoder.decode_chunk_size == params.decode_chunk_len // 2, (
            encoder.decode_chunk_size,
            params.decode_chunk_len,
        )
        T = params.decode_chunk_len + 7

        x = torch.zeros(1, T, 80, dtype=torch.float32)
        x_lens = torch.full((1,), T, dtype=torch.int32)
        states = encoder.get_init_state(device=x.device)
        encoder.__class__.forward = encoder.__class__.streaming_forward
        traced_encoder = torch.jit.trace(encoder, (x, x_lens, states))

        states1 = encoder.get_init_state(device=x.device)
        states2 = traced_encoder.get_init_state(device=x.device)
        for i in range(5):
            x = torch.randn(1, T, 80, dtype=torch.float32)
            x_lens = torch.full((1,), T, dtype=torch.int32)
            y1, _, states1 = encoder.streaming_forward(x, x_lens, states1)
            y2, _, states2 = traced_encoder(x, x_lens, states2)
            assert torch.allclose(y1, y2, atol=1e-6), (i, (y1 - y2).abs().mean())

    # Test decoder
    def _test_decoder():
        decoder = model.decoder
        y = torch.zeros(10, decoder.context_size, dtype=torch.int64)
        need_pad = torch.tensor([False])

        traced_decoder = torch.jit.trace(decoder, (y, need_pad))
        d1 = decoder(y, need_pad)
        d2 = traced_decoder(y, need_pad)
        assert torch.equal(d1, d2), (d1 - d2).abs().mean()

    # Test joiner
    def _test_joiner():
        joiner = model.joiner
        encoder_out_dim = joiner.encoder_proj.weight.shape[1]
        decoder_out_dim = joiner.decoder_proj.weight.shape[1]
        encoder_out = torch.rand(1, encoder_out_dim, dtype=torch.float32)
        decoder_out = torch.rand(1, decoder_out_dim, dtype=torch.float32)

        traced_joiner = torch.jit.trace(joiner, (encoder_out, decoder_out))
        j1 = joiner(encoder_out, decoder_out)
        j2 = traced_joiner(encoder_out, decoder_out)
        assert torch.equal(j1, j2), (j1 - j2).abs().mean()

    _test_encoder()
    _test_decoder()
    _test_joiner()


def main():
    test_model_small()
    test_model_jit_trace()


if __name__ == "__main__":
    main()
