# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../LICENSE for clarification regarding multiple authors
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

from collections import defaultdict
from typing import List, Optional, Tuple

from icefall.utils import is_module_available


class BiasedNgramLm:
    def __init__(
        self,
        backoff_id: int,
        fst = None,
        fst_filename: str = None,
        is_binary: bool = False,
    ):
        """
        Args:
          fst_filename:
            Path to the FST.
          backoff_id:
            ID of the backoff symbol.
          is_binary:
            True if the given file is a binary FST.
        """
        if not is_module_available("kaldifst"):
            raise ValueError("Please 'pip install kaldifst' first.")

        import kaldifst

        if fst is None:
            if is_binary:
                lm = kaldifst.StdVectorFst.read(fst_filename)
            else:
                with open(fst_filename, "r") as f:
                    lm = kaldifst.compile(f.read(), acceptor=False)
        else:
            lm = fst

        if not lm.is_ilabel_sorted:
            kaldifst.arcsort(lm, sort_type="ilabel")

        self.lm = lm
        self.backoff_id = backoff_id

    def _get_next_state_and_bonus_without_backoff(
        self, state: int, label: int
    ) -> Tuple[int, float]:
        """TODO: Add doc."""
        import kaldifst

        arc_iter = kaldifst.ArcIterator(self.lm, state)
        num_arcs = self.lm.num_arcs(state)

        # The LM is arc sorted by ilabel, so we use binary search below.
        left = 0
        right = num_arcs - 1
        while left <= right:
            mid = (left + right) // 2
            arc_iter.seek(mid)
            arc = arc_iter.value
            if arc.ilabel < label:
                left = mid + 1
            elif arc.ilabel > label:
                right = mid - 1
            else:
                next_states = [arc.nextstate]
                next_bonus = [arc.weight.value]

                # Handle the non-deterministic case:
                # There can be multiple out-arcs with
                # the same label.
                i = mid
                while i > 0:
                    i -= 1
                    arc_iter.seek(i)
                    arc = arc_iter.value
                    if arc.ilabel == label:
                        next_states += [arc.nextstate]
                        next_bonus += [arc.weight.value]
                    else:
                        break

                i = mid
                while i < num_arcs - 1:
                    i += 1
                    arc_iter.seek(i)
                    arc = arc_iter.value
                    if arc.ilabel == label:
                        next_states += [arc.nextstate]
                        next_bonus += [arc.weight.value]
                    else:
                        break

                return next_states, next_bonus

        return [], []

    def get_next_state_and_bonus(
        self,
        state: int,
        label: int,
    ) -> Tuple[List[int], List[float]]:

        next_states, next_bonus = self._get_next_state_and_bonus_without_backoff(
            state=state,
            label=label,
        )
        if len(next_states) > 0:
            return next_states, next_bonus

        next_states, next_bonus = self._get_next_state_and_bonus_without_backoff(
            state=state,
            label=self.backoff_id,
        )
        if len(next_states) > 0:
            assert (
                len(next_states) == 1 and next_states[0] == 0
            ), f"state={state} label={label} next_state={next_states} is not the start state"  # back-off to the start state
            return next_states, next_bonus
        else:
            return [], []


class BiasedNgramLmStateBonus:
    def __init__(self, ngram_lm: BiasedNgramLm, state_bonus: Optional[dict] = None):
        if ngram_lm is not None:
            assert ngram_lm.lm.start == 0, ngram_lm.lm.start

        self.ngram_lm = ngram_lm
        if state_bonus is not None:
            self.state_bonus = state_bonus
        else:
            self.state_bonus = defaultdict(lambda: 0)

            # At the very beginning, we are at the start state with bonus 0
            self.state_bonus[0] = 0.0

    def forward_one_step(self, label: int) -> "BiasedNgramLmStateBonus":
        if self.ngram_lm is None:
            return self

        state_bonus = defaultdict(lambda: 0)
        state_bonus[0] = self.state_bonus[
            0
        ]  # you can always stay at state 0 without any bonus or penalty
        for s, b in self.state_bonus.items():
            next_states, next_bonus = self.ngram_lm.get_next_state_and_bonus(
                s,
                label,
            )
            # print(s, b, f"={label}=>", next_state, next_bonus)
            for nc, nb in zip(next_states, next_bonus):
                state_bonus[nc] = max(state_bonus[nc], b + nb)

        return BiasedNgramLmStateBonus(ngram_lm=self.ngram_lm, state_bonus=state_bonus)

    @property
    def lm_score(self) -> float:
        if len(self.state_bonus) == 0:
            return 0

        return max(self.state_bonus.values())