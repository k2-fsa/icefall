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


class NgramLm:
    def __init__(
        self,
        fst_filename: str,
        backoff_id: int,
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

        if is_binary:
            lm = kaldifst.StdVectorFst.read(fst_filename)
        else:
            with open(fst_filename, "r") as f:
                lm = kaldifst.compile(f.read(), acceptor=False)

        if not lm.is_ilabel_sorted:
            kaldifst.arcsort(lm, sort_type="ilabel")

        self.lm = lm
        self.backoff_id = backoff_id

    def _process_backoff_arcs(
        self,
        state: int,
        cost: float,
    ) -> List[Tuple[int, float]]:
        """Similar to ProcessNonemitting() from Kaldi, this function
        returns the list of states reachable from the given state via
        backoff arcs.

        Args:
          state:
            The input state.
          cost:
            The cost of reaching the given state from the start state.
        Returns:
          Return a list, where each element contains a tuple with two entries:
            - next_state
            - cost of next_state
          If there is no backoff arc leaving the input state, then return
          an empty list.
        """
        ans = []

        next_state, next_cost = self._get_next_state_and_cost_without_backoff(
            state=state,
            label=self.backoff_id,
        )
        if next_state is None:
            return ans
        ans.append((next_state, next_cost + cost))
        ans += self._process_backoff_arcs(next_state, next_cost + cost)
        return ans

    def _get_next_state_and_cost_without_backoff(
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
                return arc.nextstate, arc.weight.value

        return None, None

    def get_next_state_and_cost(
        self,
        state: int,
        label: int,
    ) -> Tuple[List[int], List[float]]:
        states = [state]
        costs = [0]

        extra_states_costs = self._process_backoff_arcs(
            state=state,
            cost=0,
        )

        for s, c in extra_states_costs:
            states.append(s)
            costs.append(c)

        next_states = []
        next_costs = []
        for s, c in zip(states, costs):
            ns, nc = self._get_next_state_and_cost_without_backoff(s, label)
            if ns:
                next_states.append(ns)
                next_costs.append(c + nc)

        return next_states, next_costs


class NgramLmStateCost:
    def __init__(self, ngram_lm: NgramLm, state_cost: Optional[dict] = None):
        assert ngram_lm.lm.start == 0, ngram_lm.lm.start
        self.ngram_lm = ngram_lm
        if state_cost is not None:
            self.state_cost = state_cost
        else:
            self.state_cost = defaultdict(lambda: float("inf"))

            # At the very beginning, we are at the start state with cost 0
            self.state_cost[0] = 0.0

    def forward_one_step(self, label: int) -> "NgramLmStateCost":
        state_cost = defaultdict(lambda: float("inf"))
        for s, c in self.state_cost.items():
            next_states, next_costs = self.ngram_lm.get_next_state_and_cost(
                s,
                label,
            )
            for ns, nc in zip(next_states, next_costs):
                state_cost[ns] = min(state_cost[ns], c + nc)

        return NgramLmStateCost(ngram_lm=self.ngram_lm, state_cost=state_cost)

    @property
    def lm_score(self) -> float:
        if len(self.state_cost) == 0:
            return float("-inf")

        return -1 * min(self.state_cost.values())
