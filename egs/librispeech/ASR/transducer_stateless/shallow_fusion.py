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
from utils import Hypothesis, HypothesisList


def shallow_fusion(
    LG: k2.Fsa,
    hyp: Hypothesis,
    tokens: torch.Tensor,
    log_probs: torch.Tensor,
    vocab_size: int,
    blank_log_prob: torch.Tensor,
) -> HypothesisList:
    """
    Args:
      LG:
        An n-gram. It should be arc sorted, deterministic, and epsilon free.
        It contains disambig IDs and back-off arcs.
      hyp:
        The current hypothesis.
      tokens:
        The possible tokens that will be expanded from the given `hyp`.
        It is a 1-D tensor of dtype torch.int32.
      log_probs:
        It contains the acoustic log probabilities of each path that
        is extended from `hyp.ys` with `tokens`.
        log_probs.shape == tokens.shape.
      vocab_size:
        Vocabulary size, including the blank symbol. We assume that
        token IDs >= vocab_size are disambig IDs (including the backoff
        symbol #0).
      blank_log_prob:
        The log_prob for the blank token at this frame. It is from
        the output of the joiner.
    Returns:
      Return new hypotheses by extending the given `hyp` with tokens in the
      given `tokens`.
    """

    row_splits = LG.arcs.row_splits(1)
    arcs = LG.arcs.values()

    state_and_scores = copy.deepcopy(hyp.ngram_state_and_scores)

    current_states = list(state_and_scores.keys())

    # Process out-going arcs with label equal to disambig tokens or #0
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
    ans = HypothesisList()

    device = log_probs.device
    for s in current_states:
        labels_begin = row_splits[s]
        labels_end = row_splits[s + 1]
        labels = LG.labels[labels_begin:labels_end].contiguous()

        if labels[-1] == -1:
            labels = labels[:-1]

        if s != 0:
            # We add a backoff arc to the start state. Otherwise,
            # all activate state may die due to out-of-Vocabulary word.
            new_hyp = Hypothesis(
                ys=hyp.ys[:],
                log_prob=hyp.log_prob + blank_log_prob,
                ngram_state_and_scores={
                    # -20 is the cost on the backoff arc to the start state.
                    # As LG.scores.min() is about -16.6, we choose -20 here.
                    # You may need to tune this value.
                    0: torch.full((1,), -20, dtype=torch.float32, device=device)
                },
            )
            ans.add(new_hyp)

        pos = torch.searchsorted(labels, tokens)
        for i in range(pos.numel()):
            if tokens[i] == 0:
                # blank ID
                new_hyp = Hypothesis(
                    ys=hyp.ys[:],
                    log_prob=hyp.log_prob + log_probs[i],
                    ngram_state_and_scores=hyp.ngram_state_and_scores,
                )
                ans.add(new_hyp)
                continue
            elif pos[i] >= labels.numel() or labels[pos[i]] != tokens[i]:
                # No out-going arcs from this state has labels
                # equal to tokens[i]
                continue

            # Found one arc

            idx = labels_begin + pos[i]
            next_state = arcs[idx][1].item()
            score = LG.scores[idx] + state_and_scores[s]
            new_hyp = Hypothesis(
                ys=hyp.ys + [tokens[i].item()],
                log_prob=hyp.log_prob + log_probs[i],
                ngram_state_and_scores={next_state: score},
            )
            ans.add(new_hyp)

    return ans
