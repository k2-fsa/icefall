# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)

# This file implements the ideas proposed by Daniel Povey.
#
# See https://github.com/k2-fsa/snowfall/issues/232 for more details
#
import logging
from typing import List, Tuple

import torch
import k2

# Note: We use `utterance` and `sequence` interchangeably in the comment


class Nbest(object):
    '''
    An Nbest object contains two fields:

        (1) fsa, its type is k2.Fsa
        (2) shape, its type is k2.RaggedShape

    The field `fsa` is an FsaVec containing a vector of **linear** FSAs.

    The field `shape` has two axes [utt][path]. `shape.dim0()` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.
    '''

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
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

    def total_scores(self) -> k2.RaggedFloat:
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
        return k2.RaggedFloat(self.shape, scores.float())

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


    def split(self, k: int, sort: bool = True) -> Tuple['Nbest', 'Nbest']:
        '''Split the paths in the Nbest into two parts, the first part is the
        first k paths for each sequence in the Nbest, the second part is the
        remaining paths.
        There may be less than k paths for the responding sequence in the part,

        If the sort flag is true, we select the top-k paths according to the
        total_scores of each path in descending order, If a utterance has less
        than k paths, then the first part will have the really number of paths
        and leaving the second part empty.

        Args:
          k:
            Number of paths in the first part of each utterance.
        Returns:
          Return a tuple of new Nbest.
        '''
        # indexes contains idx01's for self.shape
        indexes = torch.arange(
            self.shape.num_elements(), dtype=torch.int32,
            device=self.shape.device
        )

        if sort:
            ragged_scores = self.total_scores()

            # ragged_scores.values()[indexes] is sorted
            indexes = k2.ragged.sort_sublist(
                ragged_scores, descending=True, need_new2old_indexes=True
            )

        ragged_indexes = k2.RaggedInt(self.shape, indexes)

        padded_indexes = k2.ragged.pad(ragged_indexes, value=-1)

        # Select the idx01's of top-k paths of each utterance
        first_indexes = padded_indexes[:, :k].flatten().contiguous()

        # Remove the padding elements
        first_indexes = first_indexes[first_indexes >= 0]

        first_fsas = k2.index_fsa(self.fsa, first_indexes)

        first_row_ids = k2.index(self.shape.row_ids(1), first_indexes)
        first_shape = k2.ragged.create_ragged_shape2(row_ids=first_row_ids)

        first_nbest = Nbest(first_fsas, first_shape)

        # Select the idx01's of remaining paths of each utterance
        second_indexes = padded_indexes[:, k:].flatten().contiguous()

        # Remove the padding elements
        second_indexes = second_indexes[second_indexes >= 0]

        second_fsas = k2.index_fsa(self.fsa, second_indexes)

        second_row_ids = k2.index(self.shape.row_ids(1), second_indexes)
        second_shape = k2.ragged.create_ragged_shape2(row_ids=second_row_ids)

        second_nbest = Nbest(second_fsas, second_shape)

        return first_nbest, second_nbest

