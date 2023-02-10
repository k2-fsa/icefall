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
    python ./zipformer_mmi/test_model.py
"""

import torch
from train import get_ctc_model, get_params


def test_model():
    params = get_params()
    params.vocab_size = 500
    params.num_encoder_layers = "2,4,3,2,4"
    #  params.feedforward_dims = "1024,1024,1536,1536,1024"
    params.feedforward_dims = "1024,1024,2048,2048,1024"
    params.nhead = "8,8,8,8,8"
    params.encoder_dims = "384,384,384,384,384"
    params.attention_dims = "192,192,192,192,192"
    params.encoder_unmasked_dims = "256,256,256,256,256"
    params.zipformer_downsampling_factors = "1,2,4,8,2"
    params.cnn_module_kernels = "31,31,31,31,31"
    model = get_ctc_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    features = torch.randn(2, 100, 80)
    feature_lengths = torch.full((2,), 100)
    model(x=features, x_lens=feature_lengths)


def main():
    test_model()


if __name__ == "__main__":
    main()
