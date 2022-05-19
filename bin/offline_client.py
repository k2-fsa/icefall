#!/usr/bin/env python3
# Copyright      2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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
A client for offline ASR recognition.
"""
import torch
import torchaudio
import websockets
import asyncio


async def main():
    test_wavs = [
        "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1089-134686-0001.wav",
        "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0001.wav",
        "/ceph-fj/fangjun/open-source-2/icefall-models/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13/test_wavs/1221-135766-0002.wav",
    ]
    async with websockets.connect("ws://localhost:6006") as websocket:
        while True:
            for test_wav in test_wavs:
                print(f"Sending {test_wav}")
                wave, sample_rate = torchaudio.load(test_wav)
                wave = wave.squeeze(0)
                num_bytes = wave.numel() * wave.element_size()
                print(f"Sending {num_bytes}, {wave.shape}")
                await websocket.send(
                    (num_bytes).to_bytes(8, "big", signed=True)
                )

                frame_size = 1048576 // 4  # max payload is 1MB
                num_sent_samples = 0
                start = 0
                while start < wave.numel():
                    end = start + frame_size
                    await websocket.send(wave.numpy().data[start:end])
                    start = end
                decoding_results = await websocket.recv()
                print(decoding_results)


if __name__ == "__main__":
    asyncio.run(main())
