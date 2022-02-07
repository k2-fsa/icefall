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
    python ./transducer_stateless/test_frame_shift.py
"""

import torch
from frame_shift import apply_frame_shift


def test_apply_frame_shift():
    features = torch.tensor(
        [
            [
                [1, 2, 5],
                [2, 6, 9],
                [3, 0, 2],
                [4, 11, 13],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 3, 9],
                [2, 5, 8],
                [3, 3, 6],
                [4, 0, 3],
                [5, 1, 2],
                [6, 6, 6],
            ],
        ]
    )
    supervision_segments = torch.tensor(
        [
            [0, 0, 4],
            [1, 0, 6],
        ],
        dtype=torch.int32,
    )
    shifted_features = apply_frame_shift(features, supervision_segments)

    # You can enable the debug statement in frame_shift.py
    # and check the resulting shifted_features. I've verified
    # manually that it is correct.
    print(shifted_features)


def main():
    test_apply_frame_shift()


if __name__ == "__main__":
    main()
