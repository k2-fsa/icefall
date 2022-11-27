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
    python ./conformer_ctc3/test_model.py
"""

import torch

from train import get_params, get_ctc_model


def test_model():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2

    params.dynamic_chunk_training = False
    params.short_chunk_size = 25
    params.num_left_chunks = 4
    params.causal_convolution = False

    model = get_ctc_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    features = torch.randn(2, 100, 80)
    feature_lengths = torch.full((2,), 100)
    model(x=features, x_lens=feature_lengths)


def test_model_streaming():
    params = get_params()
    params.vocab_size = 500
    params.blank_id = 0
    params.context_size = 2
    params.unk_id = 2

    params.dynamic_chunk_training = True
    params.short_chunk_size = 25
    params.num_left_chunks = 4
    params.causal_convolution = True

    model = get_ctc_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of model parameters: {num_param}")

    features = torch.randn(2, 100, 80)
    feature_lengths = torch.full((2,), 100)
    encoder_out, _ = model.encoder(x=features, x_lens=feature_lengths)
    model.get_ctc_output(encoder_out)


def main():
    test_model()
    test_model_streaming()


if __name__ == "__main__":
    main()
