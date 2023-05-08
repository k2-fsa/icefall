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

from typing import Dict, List, Tuple


class ContextState:
    """The state in ContextGraph"""

    def __init__(self, token: int, score: float, total_score: float, is_end: bool):
        """Create a ContextState.

        Args:
          token:
            The token id.
          score:
            The bonus for each token during decoding, which will hopefully
            boost the token up to survive beam search.
          total_score:
            The accumulated bonus from root of graph to current node, it will be
            used to calculate the score for fail arc.
          is_end:
            True if current token is the end of a context.
        """
        self.token = token
        self.score = score
        self.total_score = total_score
        self.is_end = is_end
        self.next = {}
        self.fail = None


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
        self.root = ContextState(token=-1, score=0, total_score=0, is_end=False)
        self.root.fail = self.root

    def _fill_fail(self):
        """This function fills the fail arc for each trie node, it can be computed
        in linear time by performing a breadth-first search starting from the root.
        See https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm for the
        details of the algorithm.
        """
        queue = []
        for token, node in self.root.next.items():
            node.fail = self.root
            queue.append(node)
        while queue:
            current_node = queue.pop(0)
            current_fail = current_node.fail
            for token, node in current_node.next.items():
                fail = current_fail
                if token in current_fail.next:
                    fail = current_fail.next[token]
                node.fail = fail
                queue.append(node)

    def build_context_graph(self, token_ids: List[List[int]]):
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
                    node.next[token] = ContextState(
                        token=token,
                        score=self.context_score,
                        # The total score is the accumulated score from root to current node,
                        # it will be used to calculate the score of fail arc later.
                        total_score=node.total_score + self.context_score,
                        is_end=i == len(tokens) - 1,
                    )
                node = node.next[token]
            self._fill_fail()

    def forward_one_step(
        self, state: ContextState, token: int
    ) -> Tuple[float, ContextState]:
        """Search the graph with given state and token.

        Args:
          state:
            The given state (trie node) to start.
          token:
            The given token.

        Returns:
          Return a tuple of score and next state.
        """
        # token matched
        if token in state.next:
            node = state.next[token]
            score = node.score
            # if the matched node is the end of a word/phrase, we will start
            # from the root for next token.
            if node.is_end:
                node = self.root
            return (score, node)
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
            # The score of the fail arc
            score = node.total_score - state.total_score
            if node.is_end:
                node = self.root
            return (score, node)

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
        score = self.root.total_score - state.total_score
        if state.is_end:
            score = 0
        return (score, self.root)


if __name__ == "__main__":
    contexts_str = ["HE", "SHE", "HIS", "HERS"]
    contexts = []
    for s in contexts_str:
        contexts.append([ord(x) for x in s])

    context_graph = ContextGraph(context_score=2)
    context_graph.build_context_graph(contexts)

    score, state = context_graph.forward_one_step(context_graph.root, ord("H"))
    assert score == 2, score
    assert state.token == ord("H"), state.token

    score, state = context_graph.forward_one_step(state, ord("I"))
    assert score == 2, score
    assert state.token == ord("I"), state.token

    score, state = context_graph.forward_one_step(state, ord("S"))
    assert score == 2, score
    assert state.token == -1, state.token

    score, state = context_graph.finalize(state)
    assert score == 0, score
    assert state.token == -1, state.token

    score, state = context_graph.forward_one_step(context_graph.root, ord("S"))
    assert score == 2, score
    assert state.token == ord("S"), state.token

    score, state = context_graph.forward_one_step(state, ord("H"))
    assert score == 2, score
    assert state.token == ord("H"), state.token

    score, state = context_graph.forward_one_step(state, ord("D"))
    assert score == -4, score
    assert state.token == -1, state.token

    score, state = context_graph.forward_one_step(context_graph.root, ord("D"))
    assert score == 0, score
    assert state.token == -1, state.token
