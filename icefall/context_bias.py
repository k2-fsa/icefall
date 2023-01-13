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

import k2
import kaldifst
from kaldifst import StdVectorFst
from kaldifst.utils import k2_to_openfst


class ContextLmStateCost:
    def __init__(
        self, context_graph: StdVectorFst, current_state=0, sum_score=0, next_state={}
    ):
        assert context_graph.start == 0, context_graph.start
        self.context_graph = context_graph
        self.current_state = current_state
        self.next_state = next_state
        self.sum_score = sum_score
        self.scores_per_tokens = None

    def is_final_state(self):
        arcs = kaldifst.ArcIterator(self.context_graph, self.current_state)
        return len(list(arcs)) == 0

    def cost_next_step(self, number_token=500, backoff_id=500):
        if self.is_final_state():
            self.sum_score = 0
            self.current_state = 0
            self.next_state = {}

        scores_per_tokens = number_token * [0.0]

        for arc in kaldifst.ArcIterator(self.context_graph, self.current_state):
            if arc.ilabel == backoff_id:
                continue

            self.next_state[arc.ilabel] = arc.nextstate
            scores_per_tokens[arc.ilabel] = -arc.weight.value

        self.scores_per_tokens = scores_per_tokens
        return scores_per_tokens

    def move_one_step(self, label: int) -> "ContextLmStateCost":
        return ContextLmStateCost(
            context_graph=self.context_graph,
            sum_score=self.sum_score + self.scores_per_tokens[label],
            current_state=self.next_state.get(label, 0),  # if no state back to start
        )


def get_context_fst(words, sp, context_score=10, backoff_id=500):
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []
    arcs_final_state = []

    for word in words:
        cur_state = loop_state
        pieces = sp.encode(word, out_type=int)
        sum_score = 0
        context_score_word = context_score / len(pieces)
        for i in range(len(pieces)):
            arcs.append(
                [cur_state, next_state, pieces[i], pieces[i], context_score_word]
            )

            sum_score += context_score_word
            arcs.append([next_state, loop_state, backoff_id, backoff_id, -sum_score])

            cur_state = next_state
            next_state += 1

        arcs_final_state.append([cur_state, -1, -1, context_score_word])

    final_state = next_state
    for final_arc in arcs_final_state:
        arcs.append([final_arc[0], final_state, final_arc[1], final_arc[2]])

    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)
    G_context = k2.Fsa.from_str(arcs, acceptor=False)

    context_fst = k2_to_openfst(G_context, olabels="aux_labels")
    return kaldifst.determinize(context_fst)
