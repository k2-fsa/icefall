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
    In GRID, the lengths of all samples are same.
    And here, we don't deploy cut operation on it.
    So, the start frame is always 0 among all samples.
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
