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
    python ./lstm_transducer_stateless/test_model.py
"""

import os
from pathlib import Path

import torch
from export import (
    export_decoder_model_jit_trace,
    export_encoder_model_jit_trace,
    export_joiner_model_jit_trace,
)
from lstm import stack_states, unstack_states
from scaling_converter import convert_scaled_to_non_scaled
from train import get_params, get_transducer_model


def test_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2
    params.encoder_dim = 512
    params.rnn_hidden_size = 1024
    params.num_encoder_layers = 12
    params.aux_layer_period = 0
    params.exp_dir = Path("exp_test_model")

    model = get_transducer_model(params)
    model.eval()

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    convert_scaled_to_non_scaled(model, inplace=True)

    if not os.path.exists(params.exp_dir):
        os.path.mkdir(params.exp_dir)

    encoder_filename = params.exp_dir / "encoder_jit_trace.pt"
    export_encoder_model_jit_trace(model.encoder, encoder_filename)

    decoder_filename = params.exp_dir / "decoder_jit_trace.pt"
    export_decoder_model_jit_trace(model.decoder, decoder_filename)

    joiner_filename = params.exp_dir / "joiner_jit_trace.pt"
    export_joiner_model_jit_trace(model.joiner, joiner_filename)

    print("The model has been successfully exported using jit.trace.")


def test_states_stack_and_unstack():
    layer, batch, hidden, cell = 12, 100, 512, 1024
    states = (
        torch.randn(layer, batch, hidden),
        torch.randn(layer, batch, cell),
    )
    states2 = stack_states(unstack_states(states))
    assert torch.allclose(states[0], states2[0])
    assert torch.allclose(states[1], states2[1])


def main():
    test_model()
    test_states_stack_and_unstack()


if __name__ == "__main__":
    main()
