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

    cd icefall/egs/aishell/ASR
    python ./pruned_transducer_stateless3/test_model.py
"""

import torch
from train import get_params, get_transducer_model


def test_model():
    params = get_params()
    params.vocab_size = 4336
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2
    params.num_encoder_layers = 36
    params.dim_feedforward = 1024
    params.nhead = 8
    params.encoder_dim = 256
    params.decoder_dim = 512
    params.joiner_dim = 512

    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")
    model.__class__.forward = torch.jit.ignore(model.__class__.forward)
    #  torch.jit.script(model)


def main():
    test_model()


if __name__ == "__main__":
    main()
