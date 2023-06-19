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

import os
import shutil
from collections import deque
from typing import Dict, List, Optional, Tuple


class ContextState:
    """The state in ContextGraph"""

    def __init__(
        self,
        id: int,
        token: int,
        token_score: float,
        node_score: float,
        local_node_score: float,
        is_end: bool,
    ):
        """Create a ContextState.

        Args:
          id:
            The node id, only for visualization now. A node is in [0, graph.num_nodes).
            The id of the root node is always 0.
          token:
            The token id.
          score:
            The bonus for each token during decoding, which will hopefully
            boost the token up to survive beam search.
          node_score:
            The accumulated bonus from root of graph to current node, it will be
            used to calculate the score for fail arc.
          local_node_score:
            The accumulated bonus from last ``end_node``(node with is_end true)
            to current_node, it will be used to calculate the score for fail arc.
            Node: The local_node_score of a ``end_node`` is 0.
          is_end:
            True if current token is the end of a context.
        """
        self.id = id
        self.token = token
        self.token_score = token_score
        self.node_score = node_score
        self.local_node_score = local_node_score
        self.is_end = is_end
        self.next = {}
        self.fail = None
        self.output = None


class ContextGraph:
    """The ContextGraph is modified from Aho-Corasick which is mainly
    a Trie with a fail arc for each node.
    See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for more details
    of Aho-Corasick algorithm.

    A ContextGraph contains some words / phrases that we expect to boost their
    scores during decoding. If the substring of a decoded sequence matches the word / phrase
    in the ContextGraph, we will give the decoded sequence a bonus to make it survive
    beam search.
    """

    def __init__(self, context_score: float):
        """Initialize a ContextGraph with the given ``context_score``.

        A root node will be created (**NOTE:** the token of root is hardcoded to -1).

        Args:
          context_score:
            The bonus score for each token(note: NOT for each word/phrase, it means longer
            word/phrase will have larger bonus score, they have to be matched though).
        """
        self.context_score = context_score
        self.num_nodes = 0
        self.root = ContextState(
            id=self.num_nodes,
            token=-1,
            token_score=0,
            node_score=0,
            local_node_score=0,
            is_end=False,
        )
        self.root.fail = self.root

    def _fill_fail_output(self):
        """This function fills the fail arc for each trie node, it can be computed
        in linear time by performing a breadth-first search starting from the root.
        See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for the
        details of the algorithm.
        """
        queue = deque()
        for token, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)
        while queue:
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                fail = current_node.fail
                if token in fail.next:
                    fail = fail.next[token]
                else:
                    fail = fail.fail
                    while token not in fail.next:
                        fail = fail.fail
                        if fail.token == -1:  # root
                            break
                    if token in fail.next:
                        fail = fail.next[token]
                node.fail = fail
                # fill the output arc
                output = node.fail
                while not output.is_end:
                    output = output.fail
                    if output.token == -1:  # root
                        output = None
                        break
                node.output = output
                queue.append(node)

    def build(self, token_ids: List[List[int]]):
        """Build the ContextGraph from a list of token list.
        It first build a trie from the given token lists, then fill the fail arc
        for each trie node.

        See https://en.wikipedia.org/wiki/Trie for how to build a trie.

        Args:
          token_ids:
            The given token lists to build the ContextGraph, it is a list of token list,
            each token list contains the token ids for a word/phrase. The token id
            could be an id of a char (modeling with single Chinese char) or an id
            of a BPE (modeling with BPEs).
        """
        for tokens in token_ids:
            node = self.root
            for i, token in enumerate(tokens):
                if token not in node.next:
                    self.num_nodes += 1
                    is_end = i == len(tokens) - 1
                    node.next[token] = ContextState(
                        id=self.num_nodes,
                        token=token,
                        token_score=self.context_score,
                        node_score=node.node_score + self.context_score,
                        local_node_score=0
                        if is_end
                        else (node.local_node_score + self.context_score),
                        is_end=is_end,
                    )
                node = node.next[token]
        self._fill_fail_output()

    def forward_one_step(
        self, state: ContextState, token: int
    ) -> Tuple[float, ContextState]:
        """Search the graph with given state and token.

        Args:
          state:
            The given token containing trie node to start.
          token:
            The given token.

        Returns:
          Return a tuple of score and next state.
        """
        node = None
        score = 0
        # token matched
        if token in state.next:
            node = state.next[token]
            score = node.token_score
            if state.is_end:
                score += state.node_score
        else:
            # token not matched
            # We will trace along the fail arc until it matches the token or reaching
            # root of the graph.
            node = state.fail
            while token not in node.next:
                node = node.fail
                if node.token == -1:  # root
                    break

            if token in node.next:
                node = node.next[token]

            # The score of the fail path
            score = node.node_score - state.local_node_score
        assert node is not None
        matched_score = 0
        output = node.output
        while output is not None:
            matched_score += output.node_score
            output = output.output
        return (score + matched_score, node)

    def finalize(self, state: ContextState) -> Tuple[float, ContextState]:
        """When reaching the end of the decoded sequence, we need to finalize
        the matching, the purpose is to subtract the added bonus score for the
        state that is not the end of a word/phrase.

        Args:
          state:
            The given state(trie node).

        Returns:
          Return a tuple of score and next state. If state is the end of a word/phrase
          the score is zero, otherwise the score is the score of a implicit fail arc
          to root. The next state is always root.
        """
        # The score of the fail arc
        score = -state.node_score
        if state.is_end:
            score = 0
        return (score, self.root)

    def draw(
        self,
        title: Optional[str] = None,
        filename: Optional[str] = "",
        symbol_table: Optional[Dict[int, str]] = None,
    ) -> "Digraph":  # noqa

        """Visualize a ContextGraph via graphviz.

        Render ContextGraph as an image via graphviz, and return the Digraph object;
        and optionally save to file `filename`.
        `filename` must have a suffix that graphviz understands, such as
        `pdf`, `svg` or `png`.

        Note:
          You need to install graphviz to use this function::

            pip install graphviz

        Args:
           title:
              Title to be displayed in image, e.g. 'A simple FSA example'
           filename:
              Filename to (optionally) save to, e.g. 'foo.png', 'foo.svg',
              'foo.png'  (must have a suffix that graphviz understands).
           symbol_table:
              Map the token ids to symbols.
        Returns:
          A Diagraph from grahpviz.
        """

        try:
            import graphviz
        except Exception:
            print("You cannot use `to_dot` unless the graphviz package is installed.")
            raise

        graph_attr = {
            "rankdir": "LR",
            "size": "8.5,11",
            "center": "1",
            "orientation": "Portrait",
            "ranksep": "0.4",
            "nodesep": "0.25",
        }
        if title is not None:
            graph_attr["label"] = title

        default_node_attr = {
            "shape": "circle",
            "style": "bold",
            "fontsize": "14",
        }

        final_state_attr = {
            "shape": "doublecircle",
            "style": "bold",
            "fontsize": "14",
        }

        final_state = -1
        dot = graphviz.Digraph(name="Context Graph", graph_attr=graph_attr)

        seen = set()
        queue = deque()
        queue.append(self.root)
        # root id is always 0
        dot.node("0", label="0", **default_node_attr)
        dot.edge("0", "0", color="red")
        seen.add(0)

        while len(queue):
            current_node = queue.popleft()
            for token, node in current_node.next.items():
                if node.id not in seen:
                    node_score = f"{node.node_score:.2f}".rstrip("0").rstrip(".")
                    local_node_score = f"{node.local_node_score:.2f}".rstrip(
                        "0"
                    ).rstrip(".")
                    label = f"{node.id}/({node_score},{local_node_score})"
                    if node.is_end:
                        dot.node(str(node.id), label=label, **final_state_attr)
                    else:
                        dot.node(str(node.id), label=label, **default_node_attr)
                    seen.add(node.id)
                weight = f"{node.token_score:.2f}".rstrip("0").rstrip(".")
                label = str(token) if symbol_table is None else symbol_table[token]
                dot.edge(str(current_node.id), str(node.id), label=f"{label}/{weight}")
                dot.edge(
                    str(node.id),
                    str(node.fail.id),
                    color="red",
                )
                if node.output is not None:
                    dot.edge(
                        str(node.id),
                        str(node.output.id),
                        color="green",
                    )
                queue.append(node)

        if filename:
            _, extension = os.path.splitext(filename)
            if extension == "" or extension[0] != ".":
                raise ValueError(
                    "Filename needs to have a suffix like .png, .pdf, .svg: {}".format(
                        filename
                    )
                )

            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_fn = dot.render(
                    filename="temp",
                    directory=tmp_dir,
                    format=extension[1:],
                    cleanup=True,
                )

                shutil.move(temp_fn, filename)

        return dot


if __name__ == "__main__":
    contexts_str = [
        "S",
        "HE",
        "SHE",
        "SHELL",
        "HIS",
        "HERS",
        "HELLO",
        "THIS",
        "THEM",
    ]
    contexts = []
    for s in contexts_str:
        contexts.append([ord(x) for x in s])

    context_graph = ContextGraph(context_score=1)
    context_graph.build(contexts)

    symbol_table = {}
    for contexts in contexts_str:
        for s in contexts:
            symbol_table[ord(s)] = s

    context_graph.draw(
        title="Graph for: " + " / ".join(contexts_str),
        filename="context_graph.pdf",
        symbol_table=symbol_table,
    )

    queries = {
        "HEHERSHE": 14,  # "HE", "HE", "HERS", "S", "SHE", "HE"
        "HERSHE": 12,  # "HE", "HERS", "S", "SHE", "HE"
        "HISHE": 9,  # "HIS", "S", "SHE", "HE"
        "SHED": 6,  # "S", "SHE", "HE"
        "HELL": 2,  # "HE"
        "HELLO": 7,  # "HE", "HELLO"
        "DHRHISQ": 4,  # "HIS", "S"
        "THEN": 2,  # "HE"
    }
    for query, expected_score in queries.items():
        total_scores = 0
        state = context_graph.root
        for q in query:
            score, state = context_graph.forward_one_step(state, ord(q))
            total_scores += score
        score, state = context_graph.finalize(state)
        assert state.token == -1, state.token
        total_scores += score
        assert total_scores == expected_score, (
            total_scores,
            expected_score,
            query,
        )
