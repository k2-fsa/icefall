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
    python ./transducer_emformer/test_emformer.py
"""

import warnings

import torch
from emformer import Emformer


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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (y_lens == ((x_lens - 1) // 2 - 1) // 2).all()
    assert x.size(0) == x.size(0)
    assert y.size(1) == max(y_lens)
    assert y.size(2) == output_dim

    num_param = sum([p.numel() for p in encoder.parameters()])
    print(f"Number of encoder parameters: {num_param}")


def test_emformer_infer_batch_single_consistency():
    """Test consistency of cached states and output logits between single
    utterance inference and batch inference."""
    from emformer import Emformer

    num_features = 80
    output_dim = 1000
    chunk_length = 8
    num_chunks = 3
    U = num_chunks * chunk_length
    L, R = 128, 4
    B, D = 2, 256
    num_encoder_layers = 4
    for use_memory in [True, False]:
        if use_memory:
            M = 3
        else:
            M = 0
        model = Emformer(
            num_features=num_features,
            output_dim=output_dim,
            segment_length=chunk_length,
            subsampling_factor=4,
            d_model=D,
            nhead=4,
            dim_feedforward=1024,
            num_encoder_layers=num_encoder_layers,
            left_context_length=L,
            right_context_length=R,
            max_memory_size=M,
            vgg_frontend=False,
        )
        model.eval()

        def save_states(states):
            saved_states = []
            for layer_idx in range(len(states)):
                layer_state = []
                layer_state.append(states[layer_idx][0].clone())  # memory
                layer_state.append(
                    states[layer_idx][1].clone()
                )  # left_context_key
                layer_state.append(
                    states[layer_idx][2].clone()
                )  # left_context_val
                layer_state.append(states[layer_idx][3].clone())  # past_length
                saved_states.append(layer_state)
            return saved_states

        def assert_states_equal(saved_states, states, sample_idx):
            for layer_idx in range(len(saved_states)):
                # assert eqaul memory
                assert torch.allclose(
                    states[layer_idx][0],
                    saved_states[layer_idx][0][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert equal left_context_key
                assert torch.allclose(
                    states[layer_idx][1],
                    saved_states[layer_idx][1][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert equal left_context_val
                assert torch.allclose(
                    states[layer_idx][2],
                    saved_states[layer_idx][2][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                    atol=1e-5,
                    rtol=0.0,
                )
                # assert eqaul past_length
                assert torch.equal(
                    states[layer_idx][3],
                    saved_states[layer_idx][3][
                        :, sample_idx : sample_idx + 1  # noqa
                    ],
                )

        x = torch.randn(B, U + R + 3, num_features)
        batch_logits = []
        batch_states = []
        states = None
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_length
            end_idx = start_idx + chunk_length
            chunk = x[:, start_idx : end_idx + R + 3]  # noqa
            lengths = torch.tensor([chunk_length + R + 3]).expand(B)
            logits, output_lengths, states = model.streaming_forward(
                chunk, lengths, states
            )
            batch_logits.append(logits)
            batch_states.append(save_states(states))
        batch_logits = torch.cat(batch_logits, dim=1)

        single_logits = []
        for sample_idx in range(B):
            sample = x[sample_idx : sample_idx + 1]  # noqa
            chunk_logits = []
            states = None
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_length
                end_idx = start_idx + chunk_length
                chunk = sample[:, start_idx : end_idx + R + 3]  # noqa
                lengths = torch.tensor([chunk_length + R + 3])
                logits, output_lengths, states = model.streaming_forward(
                    chunk, lengths, states
                )
                chunk_logits.append(logits)

                assert_states_equal(batch_states[chunk_idx], states, sample_idx)

            chunk_logits = torch.cat(chunk_logits, dim=1)
            single_logits.append(chunk_logits)
        single_logits = torch.cat(single_logits, dim=0)

        assert torch.allclose(batch_logits, single_logits, atol=1e-5, rtol=0.0)


def main():
    test_emformer()
    test_emformer_infer_batch_single_consistency()


if __name__ == "__main__":
    torch.manual_seed(20220329)
    main()
