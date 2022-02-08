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

from typing import Dict

import k2
import torch


def shallow_fusion(
    LG: k2.Fsa,
    token: int,
    state_and_scores: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    Args:
      LG:
        An n-gram. It should be arc sorted and epsilon free.
      token:
        The input token ID.
      state_and_scores:
        The keys contain the current state we are in and the
        values are the LM log_prob for reaching the corresponding
        states from the start state.
    Returns:
      Return a new state_and_scores.
    """
    row_splits = LG.arcs.row_splits(1)
    arcs = LG.arcs.values()

    current_states = list(state_and_scores.keys())

    ans = dict()
    for s in current_states:
        labels_begin = row_splits[s]
        labels_end = row_splits[s + 1]
        labels = LG.labels[labels_begin:labels_end].contiguous()

        # As LG is not deterministic, there may be multiple
        # out-going arcs that with label equal to "token"
        #
        # Note: LG is arc sorted!
        left = torch.bucketize(token, labels, right=False)
        right = torch.bucketize(token, labels, right=True)

        if left >= right:
            # There are no out-going arcs from this state
            # that have label equal to "token"
            continue

        # Now we have
        #  labels[i] == token
        # for
        #  left <= i < right

        for i in range(left, right):
            i += labels_begin
            next_state = arcs[i][1].item()
            score = LG.scores[i]
            if next_state not in ans:
                ans[next_state] = score
            else:
                ans[next_state] = max(score, ans[next_state])

    return ans
