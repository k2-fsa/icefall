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

import torch
from lhotse.utils import LOG_EPSILON


def apply_frame_shift(
    features: torch.Tensor,
    supervision_segments: torch.Tensor,
) -> torch.Tensor:
    """Apply random frame shift along the time axis.

    For instance, for the input frame `[a, b, c, d]`,

        - If frame shift is 0, the resulting output is `[a, b, c, d]`
        - If frame shift is -1, the resulting output is `[b, c, d, a]`
        - If frame shift is 1, the resulting output is `[d, a, b, c]`
        - If frame shift is 2, the resulting output is `[c, d, a, b]`

    Args:
      features:
        A 3-D tensor of shape (N, T, C).
      supervision_segments:
        A 2-D tensor of shape (num_seqs, 3). The first column is
        `sequence_idx`, the second column is `start_frame`, and
        the third column is `num_frames`.
    Returns:
      Return a 3-D tensor of shape (N, T, C).
    """
    # We assume the subsampling_factor is 4. If you change the
    # subsampling_factor, you should also change the following
    # list accordingly
    #
    # The value in frame_shifts is selected in such a way that
    # "value % subsampling_factor" is not duplicated in frame_shifts.
    frame_shifts = [-1, 0, 1, 2]

    N = features.size(0)

    # We don't support cut concatenation here
    assert torch.all(
        torch.eq(supervision_segments[:, 0], torch.arange(N))
    ), supervision_segments

    ans = []
    for i in range(N):
        start = supervision_segments[i, 1]
        end = start + supervision_segments[i, 2]

        feat = features[i, start:end, :]

        r = torch.randint(low=0, high=len(frame_shifts), size=(1,)).item()
        frame_shift = frame_shifts[r]

        # You can enable the following debug statement
        # and run ./transducer_stateless/test_frame_shift.py to
        # view the debug output.
        #  print("frame_shift", frame_shift)

        feat = torch.roll(feat, shifts=frame_shift, dims=0)
        ans.append(feat)

    ans = torch.nn.utils.rnn.pad_sequence(
        ans,
        batch_first=True,
        padding_value=LOG_EPSILON,
    )
    assert features.shape == ans.shape

    return ans
