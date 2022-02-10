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
import copy


def shallow_fusion(
    LG: k2.Fsa,
    token: int,
    state_and_scores: Dict[int, torch.Tensor],
    vocab_size: int,
) -> Dict[int, torch.Tensor]:
    """
    Args:
      LG:
        An n-gram. It should be arc sorted, deterministic, and epsilon free.
      token:
        The input token ID.
      state_and_scores:
        The keys contain the current state we are in and the
        values are the LM log_prob for reaching the corresponding
        states from the start state.
      vocab_size:
        Vocabulary size, including the blank symbol. We assume that
        token IDs >= vocab_size are disambig IDs (including the backoff
        symbol #0).
    Returns:
      Return a new state_and_scores.
    """
    row_splits = LG.arcs.row_splits(1)
    arcs = LG.arcs.values()

    state_and_scores = copy.deepcopy(state_and_scores)

    current_states = list(state_and_scores.keys())

    # Process out-going arcs with label being disambig tokens and #0
    while len(current_states) > 0:
        s = current_states.pop()
        labels_begin = row_splits[s]
        labels_end = row_splits[s + 1]
        labels = LG.labels[labels_begin:labels_end].contiguous()

        for i in reversed(range(labels.numel())):
            lab = labels[i]
            if lab == -1:
                # Note: When sorting arcs, k2 treats arc labels as
                # unsigned types
                continue

            if lab < vocab_size:
                # Since LG is arc sorted, we can exit
                # the for loop as soon as we have a label
                # with ID less than vocab_size
                break

            # This is a diambig token or #0
            idx = labels_begin + i
            next_state = arcs[idx][1].item()
            score = LG.scores[idx] + state_and_scores[s]
            if next_state not in state_and_scores:
                state_and_scores[next_state] = score
                current_states.append(next_state)
            else:
                state_and_scores[next_state] = max(
                    score, state_and_scores[next_state]
                )

    current_states = list(state_and_scores.keys())
    ans = dict()
    for s in current_states:
        labels_begin = row_splits[s]
        labels_end = row_splits[s + 1]
        labels = LG.labels[labels_begin:labels_end].contiguous()

        if labels[-1] == -1:
            labels = labels[:-1]

        pos = torch.searchsorted(labels, token)
        if pos >= labels.numel() or labels[pos] != token:
            continue

        idx = labels_begin + pos
        next_state = arcs[idx][1].item()
        score = LG.scores[idx] + state_and_scores[s]

        if next_state not in ans:
            ans[next_state] = score
        else:
            ans[next_state] = max(score, ans[next_state])

    return ans
