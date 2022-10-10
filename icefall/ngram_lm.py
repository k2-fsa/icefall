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

from typing import List, Tuple

import kaldifst


class NgramLm:
    def __init__(
        self,
        binary_fst_filename: str,
        backoff_id: int,
    ):
        """
        Args:
          binary_fst_filename:
            Path to the binary FST.
          backoff_id:
            ID of the backoff symbol.
        """
        lm = kaldifst.StdVectorFst.read(binary_fst_filename)
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
