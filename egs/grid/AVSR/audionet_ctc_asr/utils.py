# Copyright      2021  Xiaomi Corp.        (authors: Mingshuang Luo)
#
# See ../../LICENSE for clarification regarding multiple authors
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

import torch


def encode_supervisions(nnet_output_shape, batch):
    """
    Encodes the output of net and texts into
    a pair of torch Tensor, and a list of transcription strings.

    The supervision tensor has shape ``(batch_size, 3)``.
    Its second dimension contains information about sequence index [0],
    start frames [1] and num frames [2].

    In GRID, the start frame of each audio sample is 0.
    """
    N, T, D = nnet_output_shape

    supervisions_idx = torch.arange(0, N).to(torch.int32)
    start_frames = [0 for _ in range(N)]
    supervisions_start_frame = torch.tensor(start_frames).to(torch.int32)
    num_frames = [T for _ in range(N)]
    supervisions_num_frames = torch.tensor(num_frames).to(torch.int32)

    supervision_segments = torch.stack(
        (
            supervisions_idx,
            supervisions_start_frame,
            supervisions_num_frames,
        ),
        1,
    ).to(torch.int32)
    texts = batch["txt"]

    return supervision_segments, texts
