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
    python ./transducer_emformer/test_streaming_feature_extractor.py
"""

import torch
from streaming_feature_extractor import Stream


def test_streaming_feature_extractor():
    stream = Stream(context_size=2, blank_id=0)
    samples = torch.rand(16000)
    start = 0
    while True:
        n = torch.randint(50, 500, (1,)).item()
        end = start + n
        this_chunk = samples[start:end]
        start = end

        if len(this_chunk) == 0:
            break
        stream.accept_waveform(sampling_rate=16000, waveform=this_chunk)
        print(len(stream.feature_frames))
    stream.input_finished()
    print(len(stream.feature_frames))


def main():
    test_streaming_feature_extractor()


if __name__ == "__main__":
    main()
