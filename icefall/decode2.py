# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

# NOTE: This file is a refactor of decode.py
# We will delete decode.py and rename this file to decode.py

import k2
import torch


class Nbest(object):
    """
    An Nbest object contains two fields:

        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
        (2) shape. Its type is :class:`k2.RaggedShape`.

    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.

    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with the `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        assert len(fsa.shape) == 3, f"fsa.shape: {fsa.shape}"
        assert shape.num_axes == 2, f"num_axes: {shape.num_axes}"

        if fsa.shape[0] != shape.tot_size(1):
            raise ValueError(
                f"{fsa.shape[0]} vs {shape.tot_size(1)}\n"
                "Number of FSAs in `fsa` does not match the given shape"
            )

        self.fsa = fsa
        self.shape = shape

    def __str__(self):
        s = "Nbest("
        s += f"num_seqs:{self.shape.dim0}, "
        s += f"num_fsas:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc].
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
          use_double_scores:
            True to use double precision in :func:`k2.random_paths`.
            False to use single precision.
          scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths with the risk being not
            to sample the path with the best score.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path, remove_axis=True)

        # Each utterance has `num_paths` paths but some of them transduces
        # to the same word sequence, so we need to remove repeated word
        # sequences within an utterance. After removing repeats, each utterance
        # contains different number of paths
        #
        # `new2old` is a 1-D torch.Tensor mapping from the output path index
        # to the input path index.
        _, _, new2old = word_seq.unique(
            need_num_repeats=False, need_new2old_indexes=True
        )

        # kept_path is a ragged tensor with dtype torch.int32.
        # It has axes [utt][path][arc_pos]
        kept_path, _ = path.index(new2old, axis=1, need_value_indexes=False)

        # utt_to_path_shape has axes [utt][path]
        utt_to_path_shape = kept_path.shape.get_layer(0)

        # Remove the utterance axis.
        # Now kept_path has only two axes [path][arc_pos]
        kept_path = kept_path.remove_axis(0)

        # labels is a ragged tensor with 2 axes [path][token_id]
        # Note that it contains -1s.
        labels = k2.ragged.index(lattice.labels.contiguous(), kept_path)

        # Remove -1 from labels as we will use it to construct a linear FSA
        labels = labels.remove_values_eq(-1)

        if isinstance(lattice.aux_labels, k2.RaggedTensor):
            # lattice.aux_labels is a ragged tensor with dtype torch.int32.
            # It has 2 axes [arc][word], so aux_labels is also a ragged tensor
            # with 2 axes [arc][word]
            aux_labels, _ = lattice.aux_labels.index(
                indexes=kept_path.data, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.data)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice and get 1-best
        path from the resulting FsaVec.

        Caution:
          We assume `self.fsa.labels` and `lattice.labels` are token IDs.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`.
        """
        assert (
            self.fsa.device == lattice.device
        ), f"{self.fsa.device} vs {lattice.device}"

        assert len(lattice.shape) == 3, f"{lattice.shape}"

        assert (
            lattice.arcs.dim0() == self.shape.dim0
        ), f"{lattice.arcs.dim0()} vs {self.shape.dim0}"

        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        # delete token IDs as it is not needed
        del word_fsa.aux_labels
        word_fsa.scores.zero_()
        word_fsa_with_epsilon_loops = k2.remove_epsilon_and_add_self_loops(
            word_fsa
        )

        path_to_utt_map = self.shape.row_ids(1)

        # lattice has token IDs as labels and word IDs as aux_labels.
        # inv_lattice has word IDs as labels and token IDs as aux_labels
        inv_lattice = k2.invert(lattice)
        inv_lattice = k2.arc_sort(inv_lattice)

        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=path_to_utt_map,
            sorted_match_a=True,
        )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(
            path_lattice, use_double_scores=use_double_scores
        )

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of the FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semirng and tropical
          semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
        """
        # Use single precision since there are only additions.
        scores = self.fsa.get_tot_scores(
            use_double_scores=False, log_semiring=False
        )
        return k2.RaggedTensor(self.shape, scores)


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    scale: float = 1.0,
) -> k2.Fsa:
    """ """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        scale=scale,
    )

    nbest = nbest.intersect(lattice)

    # max_indexes contains the indexes for the max scores
    # of paths within an utterance.
    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path
