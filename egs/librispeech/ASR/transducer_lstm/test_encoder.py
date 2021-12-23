#!/usr/bin/env python3
#
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
    python ./transducer_lstm/test_encoder.py
"""

from encoder import LstmEncoder


def test_encoder():
    encoder = LstmEncoder(
        num_features=80,
        hidden_size=1024,
        proj_size=512,
        output_dim=512,
        subsampling_factor=4,
        num_encoder_layers=12,
    )
    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(num_params)
    # 93979284
    # 66427392


def main():
    test_encoder()


if __name__ == "__main__":
    main()
