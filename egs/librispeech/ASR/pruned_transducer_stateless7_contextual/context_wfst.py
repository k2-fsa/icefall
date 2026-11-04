import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import k2
import sentencepiece as spm
import torch
from kaldifst.utils import k2_to_openfst

def generate_context_graph_simple(
    words_pieces_list: list,
    backoff_id: int,
    sp: spm.SentencePieceProcessor,
    bonus_per_token: float = 0.1,
):
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    This context graph is a WFST as in
    `https://arxiv.org/abs/1808.02480`
    or
    `https://wenet.org.cn/wenet/context.html`.
    It is simple, as it does not have the capability to detect
    word boundaries. So, if a biasing word (e.g., 'us', the country)
    happens to be the prefix of another word (e.g., 'useful'),
    the word 'useful' will be mistakenly boosted. This is not desired.
    However, this context graph is easy to understand.

    Args:
      words_pieces_list:
        A list (batch) of lists. Each sub-list contains the context for
        the utterance. The sub-list again is a list of lists. Each sub-sub-list
        is the token sequence of a word.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.
    
    Returns:
      Return an `openfst` object representing the context graph.
    """
    # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.
    flip = -1

    fsa_list = []
    fsa_sizes = []
    for words_pieces in words_pieces_list:
        start_state = 0
        next_state = 1  # the next un-allocated state, will be incremented as we go.
        arcs = []
        
        arcs.append([start_state, start_state, backoff_id, 0, 0.0])
        # for token_id in range(sp.vocab_size()):
        #     arcs.append([start_state, start_state, token_id, 0, 0.0])

        for tokens in words_pieces:
            assert len(tokens) > 0
            cur_state = start_state

            for i in range(len(tokens) - 1):
                arcs.append(
                    [
                        cur_state, 
                        next_state, 
                        tokens[i], 
                        0, 
                        flip * bonus_per_token
                    ]
                )
                arcs.append(
                    [
                        next_state,
                        start_state,
                        backoff_id,
                        0,
                        flip * -bonus_per_token * (i + 1),
                    ]
                )

                cur_state = next_state
                next_state += 1

            # now for the last token of this word
            i = len(tokens) - 1
            arcs.append([cur_state, start_state, tokens[i], 0, flip * bonus_per_token])

        final_state = next_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        fsa = k2.arc_sort(fsa)
        fsa_sizes.append((fsa.shape[0], fsa.num_arcs))  # (n_states, n_arcs)

        fsa = k2_to_openfst(fsa, olabels="aux_labels")
        fsa_list.append(fsa)
    
    return fsa_list, fsa_sizes


def generate_context_graph_nfa(
    words_pieces_list: list,
    backoff_id: int,
    sp: spm.SentencePieceProcessor,
    bonus_per_token: float = 0.1,
):
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    This context graph is a WFST capable of detecting word boundaries.
    It is epsilon-free, non-deterministic.

    Args:
      words_pieces_list:
        A list (batch) of lists. Each sub-list contains the context for
        the utterance. The sub-list again is a list of lists. Each sub-sub-list
        is the token sequence of a word.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.
    
    Returns:
      Return an `openfst` object representing the context graph.
    """
    # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.
    flip = -1

    fsa_list = []
    fsa_sizes = []
    for words_pieces in words_pieces_list:
        start_state = 0
        # if the path go through this state, then a word boundary is detected
        boundary_state = 1
        next_state = 2  # the next un-allocated state, will be incremented as we go.
        arcs = []

        # arcs.append([start_state, start_state, backoff_id, 0, 0.0])
        for token_id in range(sp.vocab_size()):
            arcs.append([start_state, start_state, token_id, 0, 0.0])
            # arcs.append([boundary_state, start_state, token_id, 0, 0.0])  # Note: adding this line here degrades performance. Why? Because it will break the ability to detect word boundary.

        for tokens in words_pieces:
            assert len(tokens) > 0
            cur_state = start_state
            
            # static/constant bonus per token
            # my_bonus_per_token = flip * bonus_per_token * biasing_list[word]
            # my_bonus_per_token = flip * 1.0 / len(tokens) * biasing_list[word]
            my_bonus_per_token = flip * bonus_per_token  # TODO: support weighted biasing list

            for i in range(len(tokens) - 1):
                arcs.append(
                    [cur_state, next_state, tokens[i], 0, my_bonus_per_token]
                )
                if i == 0:
                    arcs.append(
                        [boundary_state, next_state, tokens[i], 0, my_bonus_per_token]
                    )

                cur_state = next_state
                next_state += 1

            # now for the last token of this word
            i = len(tokens) - 1
            arcs.append(
                [cur_state, boundary_state, tokens[i], 0, my_bonus_per_token]
            )
        
        for token_id in range(sp.vocab_size()):
            token = sp.id_to_piece(token_id)
            if token.startswith("▁"):
                arcs.append([boundary_state, start_state, token_id, 0, 0.0])

        final_state = next_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([boundary_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        fsa = k2.arc_sort(fsa)
        # fsa = k2.determinize(fsa)  # No weight pushing is needed.
        fsa_sizes.append((fsa.shape[0], fsa.num_arcs))  # (n_states, n_arcs)

        fsa = k2_to_openfst(fsa, olabels="aux_labels")
        fsa_list.append(fsa)
    
    return fsa_list, fsa_sizes


class TrieNode:
    def __init__(self, token):
        self.token = token
        self.is_end = False
        self.state_id = None
        self.weight = -1e9
        self.children = {}


class Trie(object):
    # https://albertauyeung.github.io/2020/06/15/python-trie.html/

    def __init__(self):
        self.root = TrieNode("")
    
    def insert(self, word_tokens, per_token_weight):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each token in the word
        # Check if there is no child containing the token, create a new child for the current node
        for token in word_tokens:
            if token in node.children:
                node = node.children[token]
                node.weight = max(per_token_weight, node.weight)
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(token)
                node.children[token] = new_node
                node = new_node
                node.weight = per_token_weight
        
        # Mark the end of a word
        node.is_end = True


def generate_context_graph_nfa_trie(
    words_pieces_list: list,
    backoff_id: int,
    sp: spm.SentencePieceProcessor,
    bonus_per_token: float = 0.1,
):
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    This context graph is a WFST capable of detecting word boundaries.
    It is epsilon-free, non-deterministic.
    It is also optimized such that the words are organized in a trie. (from Kang Wei)
    However, this trie may not be good for having a seperate score for each word? Or not?

    Args:
      words_pieces_list:
        A list (batch) of lists. Each sub-list contains the context for
        the utterance. The sub-list again is a list of lists. Each sub-sub-list
        is the token sequence of a word.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.
    
    Returns:
      Return an `openfst` object representing the context graph.
    """
    # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.
    flip = -1

    fsa_list = []  # These are the fsas in a batch
    fsa_sizes = []
    for words_pieces in words_pieces_list:
        start_state = 0
        # if the path go through this state, then a word boundary is detected
        boundary_state = 1
        next_state = 2  # the next un-allocated state, will be incremented as we go.
        arcs = []

        # arcs.append([start_state, start_state, backoff_id, 0, 0.0])
        for token_id in range(sp.vocab_size()):
            arcs.append([start_state, start_state, token_id, 0, 0.0])
        
        # First, compile the word list into a trie (prefix tree)
        trie = Trie()
        for tokens in words_pieces:
            
            # static/constant bonus per token
            # my_bonus_per_token = flip * bonus_per_token * biasing_list[word]
            # my_bonus_per_token = flip * 1.0 / len(tokens) * biasing_list[word]
            my_bonus_per_token = flip * bonus_per_token  # TODO: support weighted biasing list

            trie.insert(tokens, my_bonus_per_token)

        trie.root.state_id = start_state
        node_list = [trie.root]
        while len(node_list) > 0:
            cur_node = node_list.pop(0)
            if cur_node.is_end:
                continue
            
            for token, child_node in cur_node.children.items():
                if child_node.state_id is None and not child_node.is_end:
                    child_node.state_id = next_state
                    next_state += 1
                
                if child_node.is_end:
                    # If the word has finished, go to the boundary state
                    arcs.append(
                        [
                            cur_node.state_id, 
                            boundary_state,
                            token, 
                            0, 
                            child_node.weight,
                        ]
                    )
                else:
                    # Otherwise, add an arc from the current node to its child node
                    arcs.append(
                        [
                            cur_node.state_id, 
                            child_node.state_id,
                            token, 
                            0, 
                            child_node.weight,
                        ]
                    )

                # If this is the first token in a word,
                # also add an arc from the boundary node to the child node
                if cur_node == trie.root:
                    arcs.append(
                        [
                            boundary_state, 
                            child_node.state_id,
                            token, 
                            0, 
                            child_node.weight,
                        ]
                    )

                node_list.append(child_node)
                
        for token_id in range(sp.vocab_size()):
            token = sp.id_to_piece(token_id)
            if token.startswith("▁"):
                arcs.append([boundary_state, start_state, token_id, 0, 0.0])

        final_state = next_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([boundary_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fsa = k2.Fsa.from_str(arcs, acceptor=False)
        fsa = k2.arc_sort(fsa)
        # fsa = k2.determinize(fsa)  # No weight pushing is needed.
        fsa_sizes.append((fsa.shape[0], fsa.num_arcs))  # (n_states, n_arcs)

        fsa = k2_to_openfst(fsa, olabels="aux_labels")
        fsa_list.append(fsa)
    
    return fsa_list, fsa_sizes
    