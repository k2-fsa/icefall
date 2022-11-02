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


from typing import List

import k2
import kaldifst
import sentencepiece as spm


class NgramLMContext:
    def __init__(
        self,
        context_words: List[str],
        sp: spm.SentencePieceProcessor,
        symbol_table: k2.SymbolTable,
        incremental_score: float = 0.0,
        context_score: float = 3.0,
    ):
        """
        Args:
          context_words:
            List of context words
          sp:
            The BPE model.
          symbol_table:
            The token symbol table.
          incremental_score:
            Incremental score added for each token
          context_score:
            Context score
        """

        # from https://github.com/wenet-e2e/wenet/blob/main/runtime/core/decoder/context_graph.cc
        contextual_fst = kaldifst.StdVectorFst()
        start_state = contextual_fst.add_state()

        contextual_fst.start = start_state
        contextual_fst.set_final(state=start_state, weight=1.0)

        for context_word in context_words:
            context_tokens = sp.encode(context_word, out_type=str)
            prev_state = start_state
            escape_score = 0.0

            for i, token in enumerate(context_tokens):
                token_id = symbol_table[token]
                score = (i * incremental_score + context_score) * len(token)

                next_state = start_state
                if i < len(context_tokens) - 1:
                    # TODO : Check implementation,
                    #  should it be i < len(context_tokens) ?
                    next_state = contextual_fst.add_state()

                contextual_fst.add_arc(
                    prev_state,
                    kaldifst.StdArc(token_id, token_id, score, next_state),
                )

                if i > 0:
                    # Adding escape arc
                    contextual_fst.add_arc(
                        prev_state,
                        kaldifst.StdArc(0, 0, -escape_score, start_state),
                    )

                prev_state = next_state
                escape_score += score

        self.contextual_fst = kaldifst.determinize(contextual_fst)

        self.start_state = start_state
        self.state_cost = {}

    def get_next_state_and_cost_context_graph(
        self, cur_state: int, token_id: int
    ):
        next_states, next_costs = [], []
        next_state = 0

        for arc in kaldifst.ArcIterator(self.contextual_fst, cur_state):
            if arc.ilabel == 0:
                # escape score, will be overwritten
                # when ilabel equals to word id.
                cost = arc.weight
            elif arc.ilabel == token_id:
                next_state = arc.next_state
                cost = arc.weight

            next_costs.append(cost)
            next_states.append(next_state)

        return next_states, next_costs
