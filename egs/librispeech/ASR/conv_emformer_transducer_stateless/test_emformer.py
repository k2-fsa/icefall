#!/usr/bin/env python3
#
# Copyright 2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                            Zengwei Yao)
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


import torch
from emformer import ConvolutionModule, Emformer, stack_states, unstack_states


def test_convolution_module_forward():
    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 3
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length,
        right_context_length,
        D,
        kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)

    utterance, right_context = conv_module(utterance, right_context)
    assert utterance.shape == (U, B, D), utterance.shape
    assert right_context.shape == (R, B, D), right_context.shape


def test_convolution_module_infer():
    from emformer import ConvolutionModule

    B, D = 2, 256
    chunk_length = 4
    right_context_length = 2
    num_chunks = 1
    U = num_chunks * chunk_length
    R = num_chunks * right_context_length
    kernel_size = 31
    conv_module = ConvolutionModule(
        chunk_length,
        right_context_length,
        D,
        kernel_size,
    )

    utterance = torch.randn(U, B, D)
    right_context = torch.randn(R, B, D)
    cache = torch.randn(B, D, kernel_size - 1)

    utterance, right_context, new_cache = conv_module.infer(
        utterance, right_context, cache
    )
    assert utterance.shape == (U, B, D), utterance.shape
    assert right_context.shape == (R, B, D), right_context.shape
    assert new_cache.shape == (B, D, kernel_size - 1), new_cache.shape


def test_state_stack_unstack():
    num_features = 80
    chunk_length = 32
    encoder_dim = 512
    num_encoder_layers = 2
    kernel_size = 31
    left_context_length = 32
    right_context_length = 8
    memory_size = 32

    model = Emformer(
        num_features=num_features,
        chunk_length=chunk_length,
        subsampling_factor=4,
        d_model=encoder_dim,
        num_encoder_layers=num_encoder_layers,
        cnn_module_kernel=kernel_size,
        left_context_length=left_context_length,
        right_context_length=right_context_length,
        memory_size=memory_size,
    )

    for batch_size in [1, 2]:
        attn_caches = [
            [
                torch.zeros(memory_size, batch_size, encoder_dim),
                torch.zeros(left_context_length // 4, batch_size, encoder_dim),
                torch.zeros(
                    left_context_length // 4,
                    batch_size,
                    encoder_dim,
                ),
            ]
            for _ in range(num_encoder_layers)
        ]
        conv_caches = [
            torch.zeros(batch_size, encoder_dim, kernel_size - 1)
            for _ in range(num_encoder_layers)
        ]
        states = [attn_caches, conv_caches]
        x = torch.randn(batch_size, 23, num_features)
        x_lens = torch.full((batch_size,), 23)
        num_processed_frames = torch.full((batch_size,), 0)
        y, y_lens, states = model.infer(
            x, x_lens, num_processed_frames=num_processed_frames, states=states
        )

        state_list = unstack_states(states)
        states2 = stack_states(state_list)

        for ss, ss2 in zip(states[0], states2[0]):
            for s, s2 in zip(ss, ss2):
                assert torch.allclose(s, s2), f"{s.sum()}, {s2.sum()}"

        for s, s2 in zip(states[1], states2[1]):
            assert torch.allclose(s, s2), f"{s.sum()}, {s2.sum()}"


def test_torchscript_consistency_infer():
    r"""Verify that scripting Emformer does not change the behavior of method `infer`."""  # noqa
    num_features = 80
    chunk_length = 32
    encoder_dim = 512
    num_encoder_layers = 2
    kernel_size = 31
    left_context_length = 32
    right_context_length = 8
    memory_size = 32
    batch_size = 2

    model = Emformer(
        num_features=num_features,
        chunk_length=chunk_length,
        subsampling_factor=4,
        d_model=encoder_dim,
        num_encoder_layers=num_encoder_layers,
        cnn_module_kernel=kernel_size,
        left_context_length=left_context_length,
        right_context_length=right_context_length,
        memory_size=memory_size,
    ).eval()
    attn_caches = [
        [
            torch.zeros(memory_size, batch_size, encoder_dim),
            torch.zeros(left_context_length // 4, batch_size, encoder_dim),
            torch.zeros(
                left_context_length // 4,
                batch_size,
                encoder_dim,
            ),
        ]
        for _ in range(num_encoder_layers)
    ]
    conv_caches = [
        torch.zeros(batch_size, encoder_dim, kernel_size - 1)
        for _ in range(num_encoder_layers)
    ]
    states = [attn_caches, conv_caches]
    x = torch.randn(batch_size, 23, num_features)
    x_lens = torch.full((batch_size,), 23)
    num_processed_frames = torch.full((batch_size,), 0)
    y, y_lens, out_states = model.infer(
        x, x_lens, num_processed_frames=num_processed_frames, states=states
    )

    sc_model = torch.jit.script(model).eval()
    sc_y, sc_y_lens, sc_out_states = sc_model.infer(
        x, x_lens, num_processed_frames=num_processed_frames, states=states
    )

    assert torch.allclose(y, sc_y)


if __name__ == "__main__":
    test_convolution_module_forward()
    test_convolution_module_infer()
    test_state_stack_unstack()
    test_torchscript_consistency_infer()
