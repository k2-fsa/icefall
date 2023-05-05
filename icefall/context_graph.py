# Copyright    2023  Xiaomi Corp.        (authors: Wei Kang)
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

from heapq import heappush, heappop
import re
from dataclasses import dataclass
from typing import List, Tuple
import argparse
import k2
import kaldifst
import sentencepiece as spm

from icefall.utils import is_module_available


class ContextGraph:
    def __init__(self, context_score: float = 1):
        self.context_score = context_score

    def build_context_graph_char(
        self, contexts: List[str], token_table: k2.SymbolTable
    ):
        """Convert a list of texts to a list-of-list of token IDs.

        Args:
          contexts:
            It is a list of strings.
            An example containing two strings is given below:

                ['你好中国', '北京欢迎您']
          token_table:
            The SymbolTable containing tokens and corresponding ids.

        Returns:
          Return a list-of-list of token IDs.
        """
        ids: List[List[int]] = []
        whitespace = re.compile(r"([ \t])")
        for text in contexts:
            text = re.sub(whitespace, "", text)
            sub_ids: List[int] = []
            skip = False
            for txt in text:
                if txt not in token_table:
                    skip = True
                    break
                sub_ids.append(token_table[txt])
            if skip:
                logging.warning(f"Skipping context {text}, as it has OOV char.")
                continue
            ids.append(sub_ids)
        self.build_context_graph(ids)

    def build_context_graph_bpe(
        self, contexts: List[str], sp: spm.SentencePieceProcessor
    ):
        contexts_bpe = sp.encode(contexts)
        self.build_context_graph(contexts_bpe)

    def build_context_graph(self, token_ids: List[List[int]]):
        graph = kaldifst.StdVectorFst()
        start_state = (
            graph.add_state()
        )  # 1st state will be state 0 (returned by add_state)
        assert start_state == 0, start_state
        graph.start = 0  # set the start state to 0
        graph.set_final(start_state, weight=kaldifst.TropicalWeight.one)

        for tokens in token_ids:
            prev_state = start_state
            next_state = start_state
            backoff_score = 0
            for i in range(len(tokens)):
                score = self.context_score
                next_state = graph.add_state() if i < len(tokens) - 1 else start_state
                graph.add_arc(
                    state=prev_state,
                    arc=kaldifst.StdArc(
                        ilabel=tokens[i],
                        olabel=tokens[i],
                        weight=score,
                        nextstate=next_state,
                    ),
                )
                if i > 0:
                    graph.add_arc(
                        state=prev_state,
                        arc=kaldifst.StdArc(
                            ilabel=0,
                            olabel=0,
                            weight=-backoff_score,
                            nextstate=start_state,
                        ),
                    )
                prev_state = next_state
                backoff_score += score
        self.graph = kaldifst.determinize(graph)
        kaldifst.arcsort(self.graph)

    def is_final_state(self, state_id: int) -> bool:
        return self.graph.final(state_id) == kaldifst.TropicalWeight.one


    def get_next_state(self, state_id: int, label: int) -> Tuple[int, float, bool]:
        arc_iter = kaldifst.ArcIterator(self.graph, state_id)
        num_arcs = self.graph.num_arcs(state_id)

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
                return (arc.nextstate, arc.weight.value, True)

        # Backoff to state 0 with the score on epsilon arc (ilabel == 0)
        arc_iter.seek(0)
        arc = arc_iter.value
        if arc.ilabel == 0:
            return (0, 0, False)
        else:
            return (0, arc.weight.value, False)


class ContextState:
    def __init__(self, graph: ContextGraph, max_states: int):
        self.graph = graph
        self.max_states = max_states
        # [(total score, (score, state_id))]
        self.states: List[Tuple[float, Tuple[float, int]]] = []

    def __str__(self):
        return ";".join([str(state) for state in self.states])

    def clone(self):
        new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
        new_context_state.states = self.states[:]
        return new_context_state

    def finalize(self) -> float:
        new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
        if len(self.states) == 0:
            return 0, new_context_state
        item = heappop(self.states)
        return item[0], new_context_state

    def forward_one_step(self, label: int) -> float:
        states = self.states[:]
        new_states = []
        # expand current label from state state
        status = self.graph.get_next_state(0, label)
        if status[2]:
            heappush(new_states, (-status[1], (status[1], status[0])))
        else:
            assert status[0] == 0 and status[2] == False, status

        # the score we have added to the path till now
        prev_max_total_score = 0
        # expand previous states with given label
        while states:
            state = heappop(states)
            if -state[0] > prev_max_total_score:
                prev_max_total_score = -state[0]

            status = self.graph.get_next_state(state[1][1], label)

            if status[2]:
                heappush(new_states, (state[0] - status[1], (status[1], status[0])))
            else:
                pass
                # assert status == (0, state[0], False), status
        num_states_drop = (
            0
            if len(new_states) <= self.max_states
            else len(new_states) - self.max_states
        )

        states = []
        if len(new_states) == 0:
            new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
            return -prev_max_total_score, new_context_state

        item = heappop(new_states)

        # if one item match a context, clear all states (means start a new context
        # from next label), and return the score of current label
        if self.graph.is_final_state(item[1][1]):
            new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
            return -item[0] - prev_max_total_score, new_context_state

        max_total_score = -item[0]
        heappush(states, item)

        while num_states_drop != 0:
            item = heappop(new_states)
            if self.graph.is_final_state(item[1][1]):
                new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
                return -item[0] - prev_max_total_score, new_context_state
            num_states_drop -= 1

        while new_states:
            item = heappop(new_states)
            if self.graph.is_final_state(item[1][1]):
                new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
                return -item[0] - prev_max_total_score, new_context_state
            heappush(states, item)
        # no context matched, the matching may continue with previous prefix,
        # or change to another prefix.
        new_context_state = ContextState(graph=self.graph, max_states=self.max_states)
        new_context_state.states = states
        return max_total_score - prev_max_total_score, new_context_state



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bpe_model",
        type=str,
        help="Path to bpe model",
    )
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)

    contexts = ["LOVE CHINA", "HELLO WORLD", "LOVE WORLD"]
    context_graph = ContextGraph()
    context_graph.build_context_graph_bpe(contexts, sp)

    if not is_module_available("graphviz"):
        raise ValueError("Please 'pip install graphviz' first.")
    import graphviz

    fst_dot = kaldifst.draw(context_graph.graph, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="context_graph.svg")
