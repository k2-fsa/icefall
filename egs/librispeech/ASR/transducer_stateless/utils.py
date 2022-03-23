# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    # Note: It contains only the acoustic part.
    log_prob: torch.Tensor

    # Used for shallow fusion
    # The key of the dict is a state index into LG
    # while the corresponding value is the LM score
    # reaching this state from the start state.
    # Note: The value tensor contains only a single entry
    # and it contains only the LM part.
    ngram_state_and_scores: Optional[Dict[int, torch.Tensor]] = None

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[str, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.

        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.

        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy

            if False:
                old_hyp.log_prob = torch.logaddexp(
                    old_hyp.log_prob, hyp.log_prob
                )
            else:
                old_hyp.log_prob = max(old_hyp.log_prob, hyp.log_prob)

            if hyp.ngram_state_and_scores is not None:
                for state, score in hyp.ngram_state_and_scores.items():
                    if (
                        state in old_hyp.ngram_state_and_scores
                        and score > old_hyp.ngram_state_and_scores[state]
                    ):
                        old_hyp.ngram_state_and_scores[state] = score
                    else:
                        old_hyp.ngram_state_and_scores[state] = score
        else:
            self._data[key] = hyp

    def get_most_probable(
        self, length_norm: bool = False, ngram_lm_scale: Optional[float] = None
    ) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
          ngram_lm_scale:
            If not None, it specifies the scale applied to the LM score.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            if ngram_lm_scale is None:
                return max(
                    self._data.values(),
                    key=lambda hyp: hyp.log_prob / len(hyp.ys),
                )
            else:
                return max(
                    self._data.values(),
                    key=lambda hyp: (
                        hyp.log_prob
                        + ngram_lm_scale
                        * max(hyp.ngram_state_and_scores.values())
                    )
                    / len(hyp.ys),
                )
        else:
            if ngram_lm_scale is None:
                return max(self._data.values(), key=lambda hyp: hyp.log_prob)
            else:
                return max(
                    self._data.values(),
                    key=lambda hyp: hyp.log_prob
                    + ngram_lm_scale * max(hyp.ngram_state_and_scores.values()),
                )

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.

        Caution:
          `self` is modified **in-place**.

        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(
        self, threshold: torch.Tensor, ngram_lm_scale: Optional[float] = None
    ) -> "HypothesisList":
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Args:
          threshold:
            Hypotheses with log_prob less than this value are removed.
          ngram_lm_scale:
            If not None, it specifies the scale applied to the LM score.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        if ngram_lm_scale is None:
            for _, hyp in self._data.items():
                if hyp.log_prob > threshold:
                    ans.add(hyp)  # shallow copy
        else:
            for _, hyp in self._data.items():
                if (
                    hyp.log_prob
                    + ngram_lm_scale * max(hyp.ngram_state_and_scores.values())
                    > threshold
                ):
                    ans.add(hyp)  # shallow copy
        return ans

    def topk(
        self, k: int, ngram_lm_scale: Optional[float] = None
    ) -> "HypothesisList":
        """Return the top-k hypothesis."""
        hyps = list(self._data.items())

        if ngram_lm_scale is None:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]
        else:
            hyps = sorted(
                hyps,
                key=lambda h: h[1].log_prob
                + ngram_lm_scale * max(h[1].ngram_state_and_scores.values()),
                reverse=True,
            )[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)
