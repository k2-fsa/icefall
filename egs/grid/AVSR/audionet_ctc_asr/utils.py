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

"""
This script is to encodes the supervisions as Tuple list.
The supervision tensor has shape ``(batch_size, 3)``.
Its second dimension contains information about sequence index [0],
start frames [1] and num frames [2].
In GRID, the start frame of each audio sample is 0.
"""
import torch


def encode_supervisions(nnet_output_shape: int, batch: dict):
    """
    Args:
      nnet_output_shape:
        The shape of nnet_output, e.g: (N, T, D).
      batch:
        A batch of dataloader, it's a dict file
        including text and aud/vid arrays.
    Return:
      The tuple list of supervisions and the text in batch.
    """
    N, T, D = nnet_output_shape

    supervisions_idx = torch.arange(0, N, dtype=torch.int32)
    supervisions_start_frame = torch.full((1, N), 0, dtype=torch.int32)[0]
    supervisions_num_frames = torch.full((1, N), T, dtype=torch.int32)[0]

    supervision_segments = torch.stack(
        (
            supervisions_idx,
            supervisions_start_frame,
            supervisions_num_frames,
        ),
        1,
    )
    texts = batch["txt"]

    return supervision_segments, texts
