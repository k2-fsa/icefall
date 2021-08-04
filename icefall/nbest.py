# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

# This file implements the ideas proposed by Daniel Povey.
#
# See https://github.com/k2-fsa/snowfall/issues/232 for more details
#
import logging
from typing import List

import torch
import _k2
import k2

# Note: We use `utterance` and `sequence` interchangeably in the comment


class Nbest(object):
    '''
    An Nbest object contains two fields:

        (1) fsa, its type is k2.Fsa
        (2) shape, its type is k2.RaggedShape (alias to _k2.RaggedShape)

    The field `fsa` is an FsaVec containing a vector of **linear** FSAs.

    The field `shape` has two axes [utt][path]. `shape.dim0()` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.
    '''

    def __init__(self, fsa: k2.Fsa, shape: _k2.RaggedShape) -> None:
        assert len(fsa.shape) == 3, f'fsa.shape: {fsa.shape}'
        assert shape.num_axes() == 2, f'num_axes: {shape.num_axes()}'

        assert fsa.shape[0] == shape.tot_size(1), \
                f'{fsa.shape[0]} vs {shape.tot_size(1)}'

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = 'Nbest('
        s += f'num_seqs:{self.shape.dim0()}, '
        s += f'num_fsas:{self.fsa.shape[0]})'
        return s

    def intersect(self, lats: k2.Fsa) -> 'Nbest':
        '''Intersect this Nbest object with a lattice and get 1-best
        path from the resulting FsaVec.

        Caution:
          We assume FSAs in `self.fsa` don't have epsilon self-loops.
          We also assume `self.fsa.labels` and `lats.labels` are token IDs.

        Args:
          lats:
            An FsaVec. It can be the return value of
            :func:`whole_lattice_rescoring`.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lats.
        '''
        assert self.fsa.device == lats.device, \
                f'{self.fsa.device} vs {lats.device}'
        assert len(lats.shape) == 3, f'{lats.shape}'
        assert lats.arcs.dim0() == self.shape.dim0(), \
                f'{lats.arcs.dim0()} vs {self.shape.dim0()}'

        lats = k2.arc_sort(lats)  # no-op if lats is already arc sorted

        fsas_with_epsilon_loops = k2.add_epsilon_self_loops(self.fsa)

        path_to_seq_map = self.shape.row_ids(1)

        ans_lats = k2.intersect_device(a_fsas=lats,
                                       b_fsas=fsas_with_epsilon_loops,
                                       b_to_a_map=path_to_seq_map,
                                       sorted_match_a=True)

        one_best = k2.shortest_path(ans_lats, use_double_scores=True)

        one_best = k2.remove_epsilon(one_best)

        return Nbest(fsa=one_best, shape=self.shape)

    def total_scores(self) -> _k2.RaggedFloat:
        '''Get total scores of the FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semirng and tropical
          semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
        '''
        scores = self.fsa.get_tot_scores(use_double_scores=True,
                                         log_semiring=False)
        # We use single precision here since we only wrap k2.RaggedFloat.
        # If k2.RaggedDouble is wrapped, we can use double precision here.
        return _k2.RaggedFloat(self.shape, scores.float())

    def top_k(self, k: int) -> 'Nbest':
        '''Get a subset of paths in the Nbest. The resulting Nbest is regular
        in that each sequence (i.e., utterance) has the same number of
        paths (k).

        We select the top-k paths according to the total_scores of each path.
        If a utterance has less than k paths, then its last path, after sorting
        by tot_scores in descending order, is repeated so that each utterance
        has exactly k paths.

        Args:
          k:
            Number of paths in each utterance.
        Returns:
          Return a new Nbest with a regular shape.
        '''
        ragged_scores = self.total_scores()

        # indexes contains idx01's for self.shape
        # ragged_scores.values()[indexes] is sorted
        indexes = k2.ragged.sort_sublist(ragged_scores,
                                         descending=True,
                                         need_new2old_indexes=True)

        ragged_indexes = k2.RaggedInt(self.shape, indexes)

        padded_indexes = k2.ragged.pad(ragged_indexes,
                                       mode='replicate',
                                       value=-1)
        assert torch.ge(padded_indexes, 0).all(), \
                'Some utterances contain empty ' \
                f'n-best: {self.shape.row_splits(1)}'

        # Select the idx01's of top-k paths of each utterance
        top_k_indexes = padded_indexes[:, :k].flatten().contiguous()

        top_k_fsas = k2.index_fsa(self.fsa, top_k_indexes)

        top_k_shape = k2.ragged.regular_ragged_shape(dim0=self.shape.dim0(),
                                                     dim1=k)
        return Nbest(top_k_fsas, top_k_shape)


def whole_lattice_rescoring(lats: k2.Fsa, G_with_epsilon_loops: k2.Fsa) -> k2.Fsa:
    '''Rescore the 1st pass lattice with an LM.

    In general, the G in HLG used to obtain `lats` is a 3-gram LM.
    This function replaces the 3-gram LM in `lats` with a 4-gram LM.

    Args:
      lats:
        The decoding lattice from the 1st pass. We assume it is the result
        of intersecting HLG with the network output.
      G_with_epsilon_loops:
        An LM. It is usually a 4-gram LM with epsilon self-loops.
        It should be arc sorted.
    Returns:
      Return a new lattice rescored with a given G.
    '''
    assert len(lats.shape) == 3, f'{lats.shape}'
    assert hasattr(lats, 'lm_scores')
    assert G_with_epsilon_loops.shape == (1, None, None), \
            f'{G_with_epsilon_loops.shape}'

    device = lats.device
    lats.scores = lats.scores - lats.lm_scores
    # Now lats contains only acoustic scores

    # We will use lm_scores from the given G, so remove lats.lm_scores here
    del lats.lm_scores
    assert hasattr(lats, 'lm_scores') is False

    # inverted_lats has word IDs as labels.
    # Its aux_labels are token IDs, which is a ragged tensor k2.RaggedInt
    # if lats.aux_labels is a ragged tensor
    inverted_lats = k2.invert(lats)
    num_seqs = lats.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)

    while True:
        try:
            rescoring_lats = k2.intersect_device(G_with_epsilon_loops,
                                                 inverted_lats,
                                                 b_to_a_map,
                                                 sorted_match_a=True)
            break
        except RuntimeError as e:
            logging.info(f'Caught exception:\n{e}\n')
            # Usually, this is an OOM exception. We reduce
            # the size of the lattice and redo k2.intersect_device()

            # NOTE(fangjun): The choice of the threshold 1e-5 is arbitrary here
            # to avoid OOM. We may need to fine tune it.
            logging.info(f'num_arcs before: {inverted_lats.num_arcs}')
            inverted_lats = k2.prune_on_arc_post(inverted_lats, 1e-5, True)
            logging.info(f'num_arcs after: {inverted_lats.num_arcs}')

    rescoring_lats = k2.top_sort(k2.connect(rescoring_lats))

    # inv_rescoring_lats has token IDs as labels
    # and word IDs as aux_labels.
    inv_rescoring_lats = k2.invert(rescoring_lats)
    return inv_rescoring_lats


def generate_nbest_list(lats: k2.Fsa, num_paths: int) -> Nbest:
    '''Generate an n-best list from a lattice.

    Args:
      lats:
        The decoding lattice from the first pass after LM rescoring.
        lats is an FsaVec. It can be the return value of
        :func:`whole_lattice_rescoring`
      num_paths:
        Size of n for n-best list. CAUTION: After removing paths
        that represent the same token sequences, the number of paths
        in different sequences may not be equal.
    Return:
      Return an Nbest object. Note the returned FSAs don't have epsilon
      self-loops.
    '''
    assert len(lats.shape) == 3

    # CAUTION: We use `phones` instead of `tokens` here because
    # :func:`compile_HLG` uses `phones`
    #
    # Note: compile_HLG is from k2-fsa/snowfall
    assert hasattr(lats, 'phones')

    assert not hasattr(lats, 'tokens')
    lats.tokens = lats.phones
    # we use tokens instead of phones in the following code

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # token_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains token IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    # Its axes are [seq][path][token_id]
    token_seqs = k2.index(lats.tokens, paths)

    # Remove epsilons (0s) and -1 from token_seqs
    token_seqs = k2.ragged.remove_values_leq(token_seqs, 0)

    # unique_token_seqs is still a k2.RaggedInt with axes [seq][path]token_id].
    # But then number of pathsin each sequence may be different.
    unique_token_seqs, _, _ = k2.ragged.unique_sequences(
        token_seqs, need_num_repeats=False, need_new2old_indexes=False)

    seq_to_path_shape = k2.ragged.get_layer(unique_token_seqs.shape(), 0)

    # Remove the seq axis.
    # Now unique_token_seqs has only two axes [path][token_id]
    unique_token_seqs = k2.ragged.remove_axis(unique_token_seqs, 0)

    token_fsas = k2.linear_fsa(unique_token_seqs)

    return Nbest(fsa=token_fsas, shape=seq_to_path_shape)
