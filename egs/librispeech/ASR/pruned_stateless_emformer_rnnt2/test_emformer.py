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
    python ./pruned_stateless_emformer_rnnt/test_emformer.py
"""

import torch
from emformer import Emformer, stack_states, unstack_states


def test_emformer():
    N = 3
    T = 300
    C = 80

    output_dim = 500

    encoder = Emformer(
        num_features=C,
        output_dim=output_dim,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=20,
        segment_length=16,
        left_context_length=120,
        right_context_length=4,
        vgg_frontend=False,
    )

    x = torch.rand(N, T, C)
    x_lens = torch.randint(100, T, (N,))
    x_lens[0] = T

    y, y_lens = encoder(x, x_lens)

    y_lens = (((x_lens - 1) >> 1) - 1) >> 1
    assert x.size(0) == x.size(0)
    assert y.size(1) == max(y_lens)
    assert y.size(2) == output_dim

    num_param = sum([p.numel() for p in encoder.parameters()])
    print(f"Number of encoder parameters: {num_param}")


def test_emformer_streaming_forward():
    N = 3
    C = 80

    output_dim = 500

    encoder = Emformer(
        num_features=C,
        output_dim=output_dim,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=20,
        segment_length=16,
        left_context_length=120,
        right_context_length=4,
        vgg_frontend=False,
    )

    x = torch.rand(N, 23, C)
    x_lens = torch.full((N,), 23)
    y, y_lens, states = encoder.streaming_forward(x=x, x_lens=x_lens)

    state_list = unstack_states(states)
    states2 = stack_states(state_list)

    for ss, ss2 in zip(states, states2):
        for s, s2 in zip(ss, ss2):
            assert torch.allclose(s, s2), f"{s.sum()}, {s2.sum()}"


def test_emformer_init_state():
    num_encoder_layers = 20
    d_model = 512
    encoder = Emformer(
        num_features=80,
        output_dim=500,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_encoder_layers=num_encoder_layers,
        segment_length=16,
        left_context_length=120,
        right_context_length=4,
        vgg_frontend=False,
    )
    init_state = encoder.get_init_state()
    assert len(init_state) == num_encoder_layers
    layer0_state = init_state[0]
    assert len(layer0_state) == 4

    assert layer0_state[0].shape == (
        0,  # max_memory_size
        1,  # batch_size
        d_model,  # input_dim
    )

    assert layer0_state[1].shape == (
        encoder.model.left_context_length,
        1,  # batch_size
        d_model,  # input_dim
    )
    assert layer0_state[2].shape == layer0_state[1].shape
    assert layer0_state[3].shape == (
        1,  # always 1
        1,  # batch_size
    )


@torch.no_grad()
def main():
    test_emformer()
    test_emformer_streaming_forward()
    test_emformer_init_state()


if __name__ == "__main__":
    torch.manual_seed(20220329)
    main()
