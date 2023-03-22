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


import logging
import re
from dataclasses import dataclass
from typing import List
import argparse
import k2
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

    def build_context_graph_char(self, contexts: List[str], token_table: k2.SymbolTable):
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
            sub_ids : List[int] = []
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

    def build_context_graph_bpe(self, contexts: List[str], sp: spm.SentencePieceProcessor):
        contexts_bpe = sp.encode(contexts)
        self.build_context_graph(contexts_bpe)

    def build_context_graph(self, token_ids: List[List[int]]):
        graph = kaldifst.StdVectorFst()
        start_state = (
            graph.add_state()
        )  # 1st state will be state 0 (returned by add_state)
        assert start_state == 0, start_state
        graph.start = 0  # set the start state to 0
        graph.set_final(start_state, weight=0)  # weight is in log space

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
    context_graph.build_context_graph_bpe(contexts, sp)

    if not is_module_available("graphviz"):
        raise ValueError("Please 'pip install graphviz' first.")
    import graphviz

    fst_dot = kaldifst.draw(context_graph.graph, acceptor=False, portrait=True)
    fst_source = graphviz.Source(fst_dot)
    fst_source.render(outfile="context_graph.svg")
