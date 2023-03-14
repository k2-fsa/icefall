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


from dataclasses import dataclass
from typing import List
import argparse
import kaldifst
import sentencepiece as spm

from icefall.utils import is_module_available


@dataclass
class ContextState:
    state_id: int = 0
    score: float = 0.0


class ContextGraph:
    def __init__(self, context_score: float = 1):
        self.context_score = context_score

    def build_context_graph(self, contexts: List[str], sp: spm.SentencePieceProcessor):

        contexts_bpe = sp.encode(contexts)
        graph = kaldifst.StdVectorFst()
        start_state = (
            graph.add_state()
        )  # 1st state will be state 0 (returned by add_state)
        assert start_state == 0, start_state
        graph.start = 0  # set the start state to 0
        graph.set_final(start_state, weight=0)  # weight is in log space

        for bpe_ids in contexts_bpe:
            prev_state = start_state
            next_state = start_state
            backoff_score = 0
            for i in range(len(bpe_ids)):
                score = self.context_score
                next_state = graph.add_state() if i < len(bpe_ids) - 1 else start_state
                graph.add_arc(
                    state=prev_state,
                    arc=kaldifst.StdArc(
                        ilabel=bpe_ids[i],
                        olabel=bpe_ids[i],
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

    def get_next_state(self, state_id: int, label: int) -> ContextState:
        next_state = 0
        score = 0
        for arc in kaldifst.ArcIterator(self.graph, state_id):
            if arc.ilabel == 0:
                score = arc.weight.value
            elif arc.ilabel == label:
                next_state = arc.nextstate
                score = arc.weight.value
                break
        return ContextState(
            state_id=next_state,
            score=score,
        )


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
    context_graph.build_context_graph(contexts, sp)

    if not is_module_available("graphviz"):
        raise ValueError("Please 'pip install graphviz' first.")
    import graphviz

    fst_dot = kaldifst.draw(context_graph.graph, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="context_graph.svg")
