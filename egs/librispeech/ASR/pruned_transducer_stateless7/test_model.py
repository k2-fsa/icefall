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
    python ./pruned_transducer_stateless7/test_model.py
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


def main():
    test_model()


if __name__ == "__main__":
    main()
