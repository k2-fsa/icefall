# Copyright    2024  (authors: Ruizhe Huang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

import k2
import kaldifst
import sentencepiece as spm
from kaldifst.utils import k2_to_openfst


def generate_context_graph_nfa(
    words_pieces_list: List[List[List[int]]],
    backoff_id: int,
    sp: spm.SentencePieceProcessor,
    bonus_per_token: float = 0.1,
) -> Tuple[List[kaldifst.StdVectorFst], List[Tuple[int, int]]]:
    """Generate the context graph (in kaldifst format) for the biasing list
    of each utterance.

    The context graph is an epsilon-free, non-deterministic WFST that can
    detect word boundaries: a biasing word is only boosted when it starts at
    a word boundary, so that, e.g., "us" does not boost "useful".

    Args:
      words_pieces_list:
        A list (batch) of lists. Each sub-list contains the biasing words of
        an utterance, each word being a list of token IDs.
      backoff_id:
        The ID of the backoff token. It serves for failure arcs.
      sp:
        The BPE model; tokens starting with "▁" start a new word.
      bonus_per_token:
        The bonus for each token of a biasing word, which helps the token
        survive the beam search.

    Returns:
      A tuple (fsa_list, fsa_sizes), with the graph of each utterance and
      its (num_states, num_arcs).
    """
    # k2_to_openfst() negates the scores, so the bonus becomes positive
    flip = -1

    fsa_list = []
    fsa_sizes = []
    for words_pieces in words_pieces_list:
        start_state = 0
        # A path through this state has just completed a word
        boundary_state = 1
        next_state = 2  # the next unallocated state
        arcs = []

        for token_id in range(sp.vocab_size()):
            arcs.append([start_state, start_state, token_id, 0, 0.0])
            # Note: also adding [boundary_state, start_state, token_id] arcs
            # here would break word boundary detection and degrade results.

        my_bonus_per_token = flip * bonus_per_token
        for tokens in words_pieces:
            assert len(tokens) > 0
            cur_state = start_state

            for i in range(len(tokens) - 1):
                arcs.append([cur_state, next_state, tokens[i], 0, my_bonus_per_token])
                if i == 0:
                    arcs.append(
                        [boundary_state, next_state, tokens[i], 0, my_bonus_per_token]
                    )
                cur_state = next_state
                next_state += 1

            # The last token of the word
            arcs.append([cur_state, boundary_state, tokens[-1], 0, my_bonus_per_token])

        for token_id in range(sp.vocab_size()):
            if sp.id_to_piece(token_id).startswith("▁"):
                arcs.append([boundary_state, start_state, token_id, 0, 0.0])

        final_state = next_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([boundary_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = "\n".join(" ".join(str(i) for i in arc) for arc in arcs)

        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        fsa = k2.arc_sort(fsa)
        fsa_sizes.append((fsa.shape[0], fsa.num_arcs))

        fsa_list.append(k2_to_openfst(fsa, olabels="aux_labels"))

    return fsa_list, fsa_sizes
