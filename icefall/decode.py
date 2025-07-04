# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                    Wei Kang)
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

import logging
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import Dict, List, Optional, Tuple, Union

import k2
import torch

from icefall.context_graph import ContextGraph, ContextState
from icefall.lm_wrapper import LmScorer
from icefall.ngram_lm import NgramLm, NgramLmStateCost
from icefall.utils import add_eos, add_sos, get_texts

DEFAULT_LM_SCALE = [
    0.01,
    0.05,
    0.08,
    0.1,
    0.3,
    0.5,
    0.6,
    0.7,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.7,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.5,
    3.0,
    4.0,
    5.0,
]


def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
) -> k2.Fsa:
    """This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.

    The arguments and return value of this function are the same as
    :func:`k2.intersect_device`.
    """
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = (num_fsas + batch_size - 1) // batch_size
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


def get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


class Nbest(object):
    """
    An Nbest object contains two fields:

        (1) fsa. It is an FsaVec containing a vector of **linear** FSAs.
                 Its axes are [path][state][arc]
        (2) shape. Its type is :class:`k2.RaggedShape`.
                   Its axes are [utt][path]

    The field `shape` has two axes [utt][path]. `shape.dim0` contains
    the number of utterances, which is also the number of rows in the
    supervision_segments. `shape.tot_size(1)` contains the number
    of paths, which is also the number of FSAs in `fsa`.

    Caution:
      Don't be confused by the name `Nbest`. The best in the name `Nbest`
      has nothing to do with `best scores`. The important part is
      `N` in `Nbest`, not `best`.
    """

    def __init__(self, fsa: k2.Fsa, shape: k2.RaggedShape) -> None:
        """
        Args:
          fsa:
            An FsaVec with axes [path][state][arc]. It is expected to contain
            a list of **linear** FSAs.
          shape:
            A ragged shape with two axes [utt][path].
        """
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
        s += f"Number of utterances:{self.shape.dim0}, "
        s += f"Number of Paths:{self.fsa.shape[0]})"
        return s

    @staticmethod
    def from_lattice(
        lattice: k2.Fsa,
        num_paths: int,
        use_double_scores: bool = True,
        nbest_scale: float = 0.5,
    ) -> "Nbest":
        """Construct an Nbest object by **sampling** `num_paths` from a lattice.

        Each sampled path is a linear FSA.

        We assume `lattice.labels` contains token IDs and `lattice.aux_labels`
        contains word IDs.

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
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
        Returns:
          Return an Nbest instance.
        """
        saved_scores = lattice.scores.clone()
        lattice.scores *= nbest_scale
        # path is a ragged tensor with dtype torch.int32.
        # It has three axes [utt][path][arc_pos]
        path = k2.random_paths(
            lattice, num_paths=num_paths, use_double_scores=use_double_scores
        )
        lattice.scores = saved_scores

        # word_seq is a k2.RaggedTensor sharing the same shape as `path`
        # but it contains word IDs. Note that it also contains 0s and -1s.
        # The last entry in each sublist is -1.
        # It axes is [utt][path][word_id]
        if isinstance(lattice.aux_labels, torch.Tensor):
            word_seq = k2.ragged.index(lattice.aux_labels, path)
        else:
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)
        word_seq = word_seq.remove_values_leq(0)

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
                indexes=kept_path.values, axis=0, need_value_indexes=False
            )
        else:
            assert isinstance(lattice.aux_labels, torch.Tensor)
            aux_labels = k2.index_select(lattice.aux_labels, kept_path.values)
            # aux_labels is a 1-D torch.Tensor. It also contains -1 and 0.

        fsa = k2.linear_fsa(labels)
        fsa.aux_labels = aux_labels
        # Caution: fsa.scores are all 0s.
        # `fsa` has only one extra attribute: aux_labels.
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice, get 1-best
        path from the resulting FsaVec, and return a new Nbest object.

        The purpose of this function is to attach scores to an Nbest.

        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume its `labels` are word IDs.
          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`. Also, its `fsa` has non-zero scores and inherits attributes
          for `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        word_fsa.scores.zero_()
        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels
            word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
        else:
            word_fsa_with_epsilon_loops = k2.linear_fst_with_self_loops(word_fsa)

        path_to_utt_map = self.shape.row_ids(1)

        if hasattr(lattice, "aux_labels"):
            # lattice has token IDs as labels and word IDs as aux_labels.
            # inv_lattice has word IDs as labels and token IDs as aux_labels
            inv_lattice = k2.invert(lattice)
            inv_lattice = k2.arc_sort(inv_lattice)
        else:
            inv_lattice = k2.arc_sort(lattice)

        if inv_lattice.shape[0] == 1:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=torch.zeros_like(path_to_utt_map),
                sorted_match_a=True,
            )
        else:
            path_lattice = _intersect_device(
                inv_lattice,
                word_fsa_with_epsilon_loops,
                b_to_a_map=path_to_utt_map,
                sorted_match_a=True,
            )

        # path_lattice has word IDs as labels and token IDs as aux_labels
        path_lattice = k2.top_sort(k2.connect(path_lattice))

        one_best = k2.shortest_path(path_lattice, use_double_scores=use_double_scores)

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]
        am_scores = self.fsa.scores - self.fsa.lm_scores
        ragged_am_scores = k2.RaggedTensor(scores_shape, am_scores.contiguous())
        tot_scores = ragged_am_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def compute_lm_scores(self) -> k2.RaggedTensor:
        """Compute LM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: acoustic scores (AM scores)
          and n-gram language model scores (LM scores).

        Caution:
          We require that ``self.fsa`` has an attribute ``lm_scores``.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_lm_scores = k2.RaggedTensor(
            scores_shape, self.fsa.lm_scores.contiguous()
        )

        tot_scores = ragged_lm_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semiring
          and tropical semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        scores_shape = self.fsa.arcs.shape().remove_axis(1)
        # scores_shape has axes [path][arc]

        ragged_scores = k2.RaggedTensor(scores_shape, self.fsa.scores.contiguous())

        tot_scores = ragged_scores.sum()

        return k2.RaggedTensor(self.shape, tot_scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa, return_ragged=True)
        return k2.levenshtein_graph(word_ids)


def one_best_decoding(
    lattice: k2.Fsa,
    use_double_scores: bool = True,
    lm_scale_list: Optional[List[float]] = None,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """Get the best path from a lattice.

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      lm_scale_list:
        A list of floats representing LM score scales.
    Return:
      An FsaVec containing linear paths.
    """
    if lm_scale_list is not None:
        ans = dict()
        saved_am_scores = lattice.scores - lattice.lm_scores
        for lm_scale in lm_scale_list:
            am_scores = saved_am_scores / lm_scale
            lattice.scores = am_scores + lattice.lm_scores

            best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
            key = f"lm_scale_{lm_scale}"
            ans[key] = best_path
        return ans

    return k2.shortest_path(lattice, use_double_scores=use_double_scores)


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    nbest_scale: float = 1.0,
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.

    The basic idea is to first extract `num_paths` paths from the given lattice,
    build a word sequence from these paths, and compute the total scores
    of the word sequence in the tropical semiring. The one with the max score
    is used as the decoding output.

    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      **randomly**, not by ranking their scores.

    Hint:
      This decoding method is for demonstration only and it does
      not produce a lower WER than :func:`one_best_decoding`.

    Args:
      lattice:
        The decoding lattice, e.g., can be the return value of
        :func:`get_lattice`. It has 3 axes [utt][state][arc].
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are removed and only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      nbest_scale:
        It's the scale applied to the `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
    Returns:
      An FsaVec containing **linear** FSAs. It axes are [utt][state][arc].
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)
    # now nbest.fsa.scores gets assigned

    # max_indexes contains the indexes for the path with the maximum score
    # within an utterance.
    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path


def nbest_oracle(
    lattice: k2.Fsa,
    num_paths: int,
    ref_texts: List[str],
    word_table: k2.SymbolTable,
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    oov: str = "<UNK>",
) -> Dict[str, List[List[int]]]:
    """Select the best hypothesis given a lattice and a reference transcript.

    The basic idea is to extract `num_paths` paths from the given lattice,
    unique them, and select the one that has the minimum edit distance with
    the corresponding reference transcript as the decoding output.

    The decoding result returned from this function is the best result that
    we can obtain using n-best decoding with all kinds of rescoring techniques.

    This function is useful to tune the value of `nbest_scale`.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
        Note: We assume its `aux_labels` contains word IDs.
      num_paths:
        The size of `n` in n-best.
      ref_texts:
        A list of reference transcript. Each entry contains space(s)
        separated words
      word_table:
        It is the word symbol table.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      oov:
        The out of vocabulary word.
    Return:
      Return a dict. Its key contains the information about the parameters
      when calling this function, while its value contains the decoding output.
      `len(ans_dict) == len(ref_texts)`
    """
    device = lattice.device

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    hyps = nbest.build_levenshtein_graphs()

    oov_id = word_table[oov]
    word_ids_list = []
    for text in ref_texts:
        word_ids = []
        for word in text.split():
            if word in word_table:
                word_ids.append(word_table[word])
            else:
                word_ids.append(oov_id)
        word_ids_list.append(word_ids)

    refs = k2.levenshtein_graph(word_ids_list, device=device)

    levenshtein_alignment = k2.levenshtein_alignment(
        refs=refs,
        hyps=hyps,
        hyp_to_ref_map=nbest.shape.row_ids(1),
        sorted_match_ref=True,
    )

    tot_scores = levenshtein_alignment.get_tot_scores(
        use_double_scores=False, log_semiring=False
    )
    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)

    max_indexes = ragged_tot_scores.argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)
    return best_path


def rescore_with_n_best_list(
    lattice: k2.Fsa,
    G: k2.Fsa,
    num_paths: int,
    lm_scale_list: List[float],
    nbest_scale: float = 1.0,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """Rescore an n-best list with an n-gram LM.
    The path with the maximum score is used as the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. It must have the following
        attributes: ``aux_labels`` and ``lm_scores``. Its labels are
        token IDs and ``aux_labels`` word IDs.
      G:
        An FsaVec containing only a single FSA. It is an n-gram LM.
      num_paths:
        Size of nbest list.
      lm_scale_list:
        A list of floats representing LM score scales.
      nbest_scale:
        Scale to be applied to ``lattice.score`` when sampling paths
        using ``k2.random_paths``.
      use_double_scores:
        True to use double precision during computation. False to use
        single precision.
    Returns:
      A dict of FsaVec, whose key is an lm_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert G.shape == (1, None, None)
    assert G.device == device
    assert hasattr(G, "aux_labels") is False

    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        try:
            nbest = Nbest.from_lattice(
                lattice=lattice,
                num_paths=num_paths,
                use_double_scores=use_double_scores,
                nbest_scale=nbest_scale,
            )
            # nbest.fsa.scores are all 0s at this point
            nbest = nbest.intersect(lattice)
            break
        except RuntimeError as e:
            logging.info(f"Caught exception:\n{e}\n")
            logging.info(f"num_paths before decreasing: {num_paths}")
            num_paths = int(num_paths / 2)
            if loop_count >= max_loop_count or num_paths <= 0:
                logging.info("Return None as the resulting lattice is too large.")
                return None
            logging.info(
                "This OOM is not an error. You can ignore it. "
                "If your model does not converge well, or --max-duration "
                "is too large, or the input sound file is difficult to "
                "decode, you will meet this exception."
            )
            logging.info(f"num_paths after decreasing: {num_paths}")
        loop_count += 1

    # Now nbest.fsa has its scores set
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()

    nbest = nbest.intersect(G)
    # Now nbest contains only lm scores
    lm_scores = nbest.tot_scores()

    ans = dict()
    for lm_scale in lm_scale_list:
        tot_scores = am_scores.values / lm_scale + lm_scores.values
        tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


def nbest_rescore_with_LM(
    lattice: k2.Fsa,
    LM: k2.Fsa,
    num_paths: int,
    lm_scale_list: List[float],
    nbest_scale: float = 1.0,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """Rescore an n-best list with an n-gram LM.
    The path with the maximum score is used as the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. It must have the following
        attributes: ``aux_labels`` and ``lm_scores``. They are both token
        IDs.
      LM:
        An FsaVec containing only a single FSA. It is one of follows:
        - LG, L is lexicon and G is word-level n-gram LM.
        - G, token-level n-gram LM.
      num_paths:
        Size of nbest list.
      lm_scale_list:
        A list of floats representing LM score scales.
      nbest_scale:
        Scale to be applied to ``lattice.score`` when sampling paths
        using ``k2.random_paths``.
      use_double_scores:
        True to use double precision during computation. False to use
        single precision.
    Returns:
      A dict of FsaVec, whose key is an lm_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert LM.shape == (1, None, None)
    assert LM.device == device

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores contains 0s

    nbest = nbest.intersect(lattice)

    # Now nbest.fsa has its scores set
    assert hasattr(nbest.fsa, "lm_scores")

    # am scores + bi-gram scores
    hp_scores = nbest.tot_scores()

    # Now start to intersect nbest with LG or G
    inv_fsa = k2.invert(nbest.fsa)
    if hasattr(LM, "aux_labels"):
        # LM is LG here
        # delete token IDs as it is not needed
        del inv_fsa.aux_labels
    inv_fsa.scores.zero_()
    inv_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(inv_fsa)
    path_to_utt_map = nbest.shape.row_ids(1)

    LM = k2.arc_sort(LM)
    path_lattice = k2.intersect_device(
        LM,
        inv_fsa_with_epsilon_loops,
        b_to_a_map=torch.zeros_like(path_to_utt_map),
        sorted_match_a=True,
    )

    # Its labels are token IDs.
    # If LM is G, its aux_labels are tokens IDs;
    # If LM is LG, its aux_labels are words IDs.
    path_lattice = k2.top_sort(k2.connect(path_lattice))
    one_best = k2.shortest_path(path_lattice, use_double_scores=use_double_scores)

    lm_scores = one_best.get_tot_scores(
        use_double_scores=use_double_scores,
        log_semiring=True,  # Note: we always use True
    )
    # If LM is LG, we might get empty paths
    lm_scores[lm_scores == float("-inf")] = -1e9

    ans = dict()
    for lm_scale in lm_scale_list:
        tot_scores = hp_scores.values / lm_scale + lm_scores
        tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


def rescore_with_whole_lattice(
    lattice: k2.Fsa,
    G_with_epsilon_loops: k2.Fsa,
    lm_scale_list: Optional[List[float]] = None,
    use_double_scores: bool = True,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """Intersect the lattice with an n-gram LM and use shortest path
    to decode.

    The input lattice is obtained by intersecting `HLG` with
    a DenseFsaVec, where the `G` in `HLG` is in general a 3-gram LM.
    The input `G_with_epsilon_loops` is usually a 4-gram LM. You can consider
    this function as a second pass decoding. In the first pass decoding, we
    use a small G, while we use a larger G in the second pass decoding.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc]. Its `aux_lables` are word IDs.
        It must have an attribute `lm_scores`.
      G_with_epsilon_loops:
        An FsaVec containing only a single FSA. It contains epsilon self-loops.
        It is an acceptor and its labels are word IDs.
      lm_scale_list:
        Optional. If none, return the intersection of `lattice` and
        `G_with_epsilon_loops`.
        If not None, it contains a list of values to scale LM scores.
        For each scale, there is a corresponding decoding result contained in
        the resulting dict.
      use_double_scores:
        True to use double precision in the computation.
        False to use single precision.
    Returns:
      If `lm_scale_list` is None, return a new lattice which is the intersection
      result of `lattice` and `G_with_epsilon_loops`.
      Otherwise, return a dict whose key is an entry in `lm_scale_list` and the
      value is the decoding result (i.e., an FsaVec containing linear FSAs).
    """
    # Nbest is not used in this function
    assert hasattr(lattice, "lm_scores")
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lattice.device
    lattice.scores = lattice.scores - lattice.lm_scores
    # We will use lm_scores from G, so remove lats.lm_scores here
    del lattice.lm_scores

    assert hasattr(G_with_epsilon_loops, "lm_scores")

    # Now, lattice.scores contains only am_scores

    # inv_lattice has word IDs as labels.
    # Its `aux_labels` is token IDs
    inv_lattice = k2.invert(lattice)
    num_seqs = lattice.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)

    # NOTE: The choice of the threshold list is arbitrary here to avoid OOM.
    # You may need to fine tune it.
    prune_th_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    prune_th_list += [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        try:
            rescoring_lattice = k2.intersect_device(
                G_with_epsilon_loops,
                inv_lattice,
                b_to_a_map,
                sorted_match_a=True,
            )
            rescoring_lattice = k2.top_sort(k2.connect(rescoring_lattice))
            break
        except RuntimeError as e:
            logging.info(f"Caught exception:\n{e}\n")
            if loop_count >= max_loop_count:
                logging.info("Return None as the resulting lattice is too large.")
                return None
            logging.info(f"num_arcs before pruning: {inv_lattice.arcs.num_elements()}")
            logging.info(
                "This OOM is not an error. You can ignore it. "
                "If your model does not converge well, or --max-duration "
                "is too large, or the input sound file is difficult to "
                "decode, you will meet this exception."
            )
            inv_lattice = k2.prune_on_arc_post(
                inv_lattice,
                prune_th_list[loop_count],
                True,
            )
            logging.info(f"num_arcs after pruning: {inv_lattice.arcs.num_elements()}")
        loop_count += 1

    # lat has token IDs as labels
    # and word IDs as aux_labels.
    lat = k2.invert(rescoring_lattice)

    if lm_scale_list is None:
        return lat

    ans = dict()
    saved_am_scores = lat.scores - lat.lm_scores
    for lm_scale in lm_scale_list:
        am_scores = saved_am_scores / lm_scale
        lat.scores = am_scores + lat.lm_scores

        best_path = k2.shortest_path(lat, use_double_scores=use_double_scores)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


def rescore_with_attention_decoder(
    lattice: k2.Fsa,
    num_paths: int,
    model: torch.nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: Optional[torch.Tensor],
    sos_id: int,
    eos_id: int,
    nbest_scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    attention_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest score is
    the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `(T, N, C)`.
      memory_key_padding_mask:
        The padding mask for memory with shape `(N, T)`.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      nbest_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        try:
            nbest = Nbest.from_lattice(
                lattice=lattice,
                num_paths=num_paths,
                use_double_scores=use_double_scores,
                nbest_scale=nbest_scale,
            )
            # nbest.fsa.scores are all 0s at this point
            nbest = nbest.intersect(lattice)
            break
        except RuntimeError as e:
            logging.info(f"Caught exception:\n{e}\n")
            logging.info(f"num_paths before decreasing: {num_paths}")
            num_paths = int(num_paths / 2)
            if loop_count >= max_loop_count or num_paths <= 0:
                logging.info("Return None as the resulting lattice is too large.")
                return None
            logging.info(
                "This OOM is not an error. You can ignore it. "
                "If your model does not converge well, or --max-duration "
                "is too large, or the input sound file is difficult to "
                "decode, you will meet this exception."
            )
            logging.info(f"num_paths after decreasing: {num_paths}")
        loop_count += 1

    # Now nbest.fsa has its scores set.
    # Also, nbest.fsa inherits the attributes from `lattice`.
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()
    ngram_lm_scores = nbest.compute_lm_scores()

    # The `tokens` attribute is set inside `compile_hlg.py`
    assert hasattr(nbest.fsa, "tokens")
    assert isinstance(nbest.fsa.tokens, torch.Tensor)

    path_to_utt_map = nbest.shape.row_ids(1).to(torch.long)
    # the shape of memory is (T, N, C), so we use axis=1 here
    expanded_memory = memory.index_select(1, path_to_utt_map)

    if memory_key_padding_mask is not None:
        # The shape of memory_key_padding_mask is (N, T), so we
        # use axis=0 here.
        expanded_memory_key_padding_mask = memory_key_padding_mask.index_select(
            0, path_to_utt_map
        )
    else:
        expanded_memory_key_padding_mask = None

    # remove axis corresponding to states.
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
    tokens = tokens.remove_values_leq(0)
    token_ids = tokens.tolist()

    if len(token_ids) == 0:
        print("Warning: rescore_with_attention_decoder(): empty token-ids")
        return None

    nll = model.decoder_nll(
        memory=expanded_memory,
        memory_key_padding_mask=expanded_memory_key_padding_mask,
        token_ids=token_ids,
        sos_id=sos_id,
        eos_id=eos_id,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if ngram_lm_scale is None:
        ngram_lm_scale_list = [0.01, 0.05, 0.08]
        ngram_lm_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        ngram_lm_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        ngram_lm_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
    else:
        ngram_lm_scale_list = [ngram_lm_scale]

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        attention_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            tot_scores = (
                am_scores.values
                + n_scale * ngram_lm_scores.values
                + a_scale * attention_scores
            )
            ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
            max_indexes = ragged_tot_scores.argmax()
            best_path = k2.index_fsa(nbest.fsa, max_indexes)

            key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}"
            ans[key] = best_path
    return ans


def rescore_with_attention_decoder_with_ngram(
    lattice: k2.Fsa,
    num_paths: int,
    attention_decoder: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    nbest_scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    attention_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest score is
    the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      attention_decoder:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      encoder_out:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `(N, T, C)`.
      encoder_out_lens:
        Length of encoder outputs, with shape of `(N,)`.
      nbest_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    max_loop_count = 10
    loop_count = 0
    while loop_count <= max_loop_count:
        try:
            nbest = Nbest.from_lattice(
                lattice=lattice,
                num_paths=num_paths,
                use_double_scores=use_double_scores,
                nbest_scale=nbest_scale,
            )
            # nbest.fsa.scores are all 0s at this point
            nbest = nbest.intersect(lattice)
            break
        except RuntimeError as e:
            logging.info(f"Caught exception:\n{e}\n")
            logging.info(f"num_paths before decreasing: {num_paths}")
            num_paths = int(num_paths / 2)
            if loop_count >= max_loop_count or num_paths <= 0:
                logging.info("Return None as the resulting lattice is too large.")
                return None
            logging.info(
                "This OOM is not an error. You can ignore it. "
                "If your model does not converge well, or --max-duration "
                "is too large, or the input sound file is difficult to "
                "decode, you will meet this exception."
            )
            logging.info(f"num_paths after decreasing: {num_paths}")
        loop_count += 1

    # Now nbest.fsa has its scores set.
    # Also, nbest.fsa inherits the attributes from `lattice`.
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()
    ngram_lm_scores = nbest.compute_lm_scores()

    # The `tokens` attribute is set inside `compile_hlg.py`
    assert hasattr(nbest.fsa, "tokens")
    assert isinstance(nbest.fsa.tokens, torch.Tensor)

    path_to_utt_map = nbest.shape.row_ids(1).to(torch.long)
    # the shape of memory is (T, N, C), so we use axis=1 here
    expanded_encoder_out = encoder_out.index_select(0, path_to_utt_map)
    expanded_encoder_out_lens = encoder_out_lens.index_select(0, path_to_utt_map)

    # remove axis corresponding to states.
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
    tokens = tokens.remove_values_leq(0)
    token_ids = tokens.tolist()

    nll = attention_decoder.nll(
        encoder_out=expanded_encoder_out,
        encoder_out_lens=expanded_encoder_out_lens,
        token_ids=token_ids,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if ngram_lm_scale is None:
        ngram_lm_scale_list = [0.01, 0.05, 0.08]
        ngram_lm_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        ngram_lm_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        ngram_lm_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
    else:
        ngram_lm_scale_list = [ngram_lm_scale]

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        attention_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            tot_scores = (
                am_scores.values
                + n_scale * ngram_lm_scores.values
                + a_scale * attention_scores
            )
            ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
            max_indexes = ragged_tot_scores.argmax()
            best_path = k2.index_fsa(nbest.fsa, max_indexes)

            key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}"
            ans[key] = best_path
    return ans


def rescore_with_attention_decoder_no_ngram(
    lattice: k2.Fsa,
    num_paths: int,
    attention_decoder: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    nbest_scale: float = 1.0,
    attention_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest score is
    the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      attention_decoder:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      encoder_out:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `(N, T, C)`.
      encoder_out_lens:
        Length of encoder outputs, with shape of `(N,)`.
      nbest_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.

    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    # path is a ragged tensor with dtype torch.int32.
    # It has three axes [utt][path][arc_pos]
    path = k2.random_paths(lattice, num_paths=num_paths, use_double_scores=True)
    # Note that labels, aux_labels and scores contains 0s and -1s.
    # The last entry in each sublist is -1.
    # The axes are [path][token_id]
    labels = k2.ragged.index(lattice.labels.contiguous(), path).remove_axis(0)
    aux_labels = k2.ragged.index(lattice.aux_labels.contiguous(), path).remove_axis(0)
    scores = k2.ragged.index(lattice.scores.contiguous(), path).remove_axis(0)

    # Remove -1 from labels as we will use it to construct a linear FSA
    labels = labels.remove_values_eq(-1)
    fsa = k2.linear_fsa(labels)
    fsa.aux_labels = aux_labels.values

    # utt_to_path_shape has axes [utt][path]
    utt_to_path_shape = path.shape.get_layer(0)
    scores = k2.RaggedTensor(utt_to_path_shape, scores.sum())

    path_to_utt_map = utt_to_path_shape.row_ids(1).to(torch.long)
    # the shape of memory is (N, T, C), so we use axis=0 here
    expanded_encoder_out = encoder_out.index_select(0, path_to_utt_map)
    expanded_encoder_out_lens = encoder_out_lens.index_select(0, path_to_utt_map)

    token_ids = aux_labels.remove_values_leq(0).tolist()

    nll = attention_decoder.nll(
        encoder_out=expanded_encoder_out,
        encoder_out_lens=expanded_encoder_out_lens,
        token_ids=token_ids,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        attention_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
        attention_scale_list += [5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()

    for a_scale in attention_scale_list:
        tot_scores = scores.values + a_scale * attention_scores
        ragged_tot_scores = k2.RaggedTensor(utt_to_path_shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(fsa, max_indexes)

        key = f"attention_scale_{a_scale}"
        ans[key] = best_path
    return ans


def rescore_with_rnn_lm(
    lattice: k2.Fsa,
    num_paths: int,
    rnn_lm_model: torch.nn.Module,
    model: torch.nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: Optional[torch.Tensor],
    sos_id: int,
    eos_id: int,
    blank_id: int,
    nbest_scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    attention_scale: Optional[float] = None,
    rnn_lm_scale: Optional[float] = None,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """This function extracts `num_paths` paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest score is
    the decoding output.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      rnn_lm_model:
        A rnn-lm model used for LM rescoring
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `(T, N, C)`.
      memory_key_padding_mask:
        The padding mask for memory with shape `(N, T)`.
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      nbest_scale:
        It's the scale applied to `lattice.scores`. A smaller value
        leads to more unique paths at the risk of missing the correct path.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.
      rnn_lm_scale:
        Optional. It specifies the scale for RNN LM scores.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each utterance in the lattice.
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # nbest.fsa.scores are all 0s at this point

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa has its scores set.
    # Also, nbest.fsa inherits the attributes from `lattice`.
    assert hasattr(nbest.fsa, "lm_scores")

    am_scores = nbest.compute_am_scores()
    ngram_lm_scores = nbest.compute_lm_scores()

    # The `tokens` attribute is set inside `compile_hlg.py`
    assert hasattr(nbest.fsa, "tokens")
    assert isinstance(nbest.fsa.tokens, torch.Tensor)

    path_to_utt_map = nbest.shape.row_ids(1).to(torch.long)
    # the shape of memory is (T, N, C), so we use axis=1 here
    expanded_memory = memory.index_select(1, path_to_utt_map)

    if memory_key_padding_mask is not None:
        # The shape of memory_key_padding_mask is (N, T), so we
        # use axis=0 here.
        expanded_memory_key_padding_mask = memory_key_padding_mask.index_select(
            0, path_to_utt_map
        )
    else:
        expanded_memory_key_padding_mask = None

    # remove axis corresponding to states.
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
    tokens = tokens.remove_values_leq(0)
    token_ids = tokens.tolist()

    if len(token_ids) == 0:
        print("Warning: rescore_with_attention_decoder(): empty token-ids")
        return None

    nll = model.decoder_nll(
        memory=expanded_memory,
        memory_key_padding_mask=expanded_memory_key_padding_mask,
        token_ids=token_ids,
        sos_id=sos_id,
        eos_id=eos_id,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    # Now for RNN LM
    sos_tokens = add_sos(tokens, sos_id)
    tokens_eos = add_eos(tokens, eos_id)
    sos_tokens_row_splits = sos_tokens.shape.row_splits(1)
    sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

    x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
    y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

    x_tokens = x_tokens.to(torch.int64)
    y_tokens = y_tokens.to(torch.int64)
    sentence_lengths = sentence_lengths.to(torch.int64)

    rnn_lm_nll = rnn_lm_model(x=x_tokens, y=y_tokens, lengths=sentence_lengths)
    assert rnn_lm_nll.ndim == 2
    assert rnn_lm_nll.shape[0] == len(token_ids)

    rnn_lm_scores = -1 * rnn_lm_nll.sum(dim=1)

    ngram_lm_scale_list = DEFAULT_LM_SCALE
    attention_scale_list = DEFAULT_LM_SCALE
    rnn_lm_scale_list = DEFAULT_LM_SCALE

    if ngram_lm_scale:
        ngram_lm_scale_list = [ngram_lm_scale]

    if attention_scale:
        attention_scale_list = [attention_scale]

    if rnn_lm_scale:
        rnn_lm_scale_list = [rnn_lm_scale]

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            for r_scale in rnn_lm_scale_list:
                tot_scores = (
                    am_scores.values
                    + n_scale * ngram_lm_scores.values
                    + a_scale * attention_scores
                    + r_scale * rnn_lm_scores
                )
                ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
                max_indexes = ragged_tot_scores.argmax()
                best_path = k2.index_fsa(nbest.fsa, max_indexes)

                key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}_rnn_lm_scale_{r_scale}"  # noqa
                ans[key] = best_path
    return ans


def ctc_greedy_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """CTC greedy search.

    Args:
         ctc_output: (batch, seq_len, vocab_size)
         encoder_out_lens: (batch,)
    Returns:
         List[List[int]]: greedy search result
    """
    batch = ctc_output.shape[0]
    index = ctc_output.argmax(dim=-1)  # (batch, seq_len)
    hyps = [
        torch.unique_consecutive(index[i, : encoder_out_lens[i]]) for i in range(batch)
    ]

    hyps = [h[h != blank_id].tolist() for h in hyps]

    return hyps


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int] = field(default_factory=list)

    # The log prob of ys that ends with blank token.
    # It contains only one entry.
    log_prob_blank: torch.Tensor = torch.zeros(1, dtype=torch.float32)

    # The log prob of ys that ends with non blank token.
    # It contains only one entry.
    log_prob_non_blank: torch.Tensor = torch.tensor(
        [float("-inf")], dtype=torch.float32
    )

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # The lm score of ys
    # May contain external LM score (including LODR score) and contextual biasing score
    # It contains only one entry
    lm_score: torch.Tensor = torch.zeros(1, dtype=torch.float32)

    # the lm log_probs for next token given the history ys
    # The number of elements should be equal to vocabulary size.
    lm_log_probs: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # LODR (N-gram LM) state
    LODR_state: Optional[NgramLmStateCost] = None

    # N-gram LM state
    Ngram_state: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None

    # This is the total score of current path, acoustic plus external LM score.
    @property
    def tot_score(self) -> torch.Tensor:
        return self.log_prob + self.lm_score

    # This is only the probability from model output (i.e External LM score not included).
    @property
    def log_prob(self) -> torch.Tensor:
        return torch.logaddexp(self.log_prob_non_blank, self.log_prob_blank)

    @property
    def key(self) -> tuple:
        """Return a tuple representation of self.ys"""
        return tuple(self.ys)

    def clone(self) -> "Hypothesis":
        return Hypothesis(
            ys=self.ys,
            log_prob_blank=self.log_prob_blank,
            log_prob_non_blank=self.log_prob_non_blank,
            timestamp=self.timestamp,
            lm_log_probs=self.lm_log_probs,
            lm_score=self.lm_score,
            state=self.state,
            LODR_state=self.LODR_state,
            Ngram_state=self.Ngram_state,
            context_state=self.context_state,
        )


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[tuple, Hypothesis]] = None) -> None:
        """
        Args:
          data:
            A dict of Hypotheses. Its key is its `value.key`.
        """
        if data is None:
            self._data = {}
        else:
            self._data = data

    @property
    def data(self) -> Dict[tuple, Hypothesis]:
        return self._data

    def add(self, hyp: Hypothesis) -> None:
        """Add a Hypothesis to `self`.
        If `hyp` already exists in `self`, its probability is updated using
        `log-sum-exp` with the existed one.
        Args:
          hyp:
            The hypothesis to be added.
        """
        key = hyp.key
        if key in self:
            old_hyp = self._data[key]  # shallow copy
            torch.logaddexp(
                old_hyp.log_prob_blank, hyp.log_prob_blank, out=old_hyp.log_prob_blank
            )
            torch.logaddexp(
                old_hyp.log_prob_non_blank,
                hyp.log_prob_non_blank,
                out=old_hyp.log_prob_non_blank,
            )
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `tot_score`.
        Args:
          length_norm:
            If True, the `tot_score` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `tot_score`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.tot_score)

    def remove(self, hyp: Hypothesis) -> None:
        """Remove a given hypothesis.
        Caution:
          `self` is modified **in-place**.
        Args:
          hyp:
            The hypothesis to be removed from `self`.
            Note: It must be contained in `self`. Otherwise,
            an exception is raised.
        """
        key = hyp.key
        assert key in self, f"{key} does not exist"
        del self._data[key]

    def filter(self, threshold: torch.Tensor) -> "HypothesisList":
        """Remove all Hypotheses whose tot_score is less than threshold.
        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.
        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `tot_score` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.tot_score > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.
        Args:
          length_norm:
            If True, the `tot_score` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].tot_score / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].tot_score, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: tuple):
        return key in self._data

    def __getitem__(self, key: tuple):
        return self._data[key]

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(str(s))


def get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
    """Return a ragged shape with axes [utt][num_hyps].
    Args:
      hyps:
        len(hyps) == batch_size. It contains the current hypothesis for
        each utterance in the batch.
    Returns:
      Return a ragged shape with 2 axes [utt][num_hyps]. Note that
      the shape is on CPU.
    """
    num_hyps = [len(h) for h in hyps]

    # torch.cumsum() is inclusive sum, so we put a 0 at the beginning
    # to get exclusive sum later.
    num_hyps.insert(0, 0)

    num_hyps = torch.tensor(num_hyps)
    row_splits = torch.cumsum(num_hyps, dim=0, dtype=torch.int32)
    ans = k2.ragged.create_ragged_shape2(
        row_splits=row_splits, cached_tot_size=row_splits[-1].item()
    )
    return ans


def _step_worker(
    log_probs: torch.Tensor,
    indexes: torch.Tensor,
    B: HypothesisList,
    beam: int = 4,
    blank_id: int = 0,
    nnlm_scale: float = 0,
    LODR_lm_scale: float = 0,
    context_graph: Optional[ContextGraph] = None,
) -> HypothesisList:
    """The worker to decode one step.
    Args:
      log_probs:
        topk log_probs of current step (i.e. the kept tokens of first pass pruning),
        the shape is (beam,)
      topk_indexes:
        The indexes of the topk_values above, the shape is (beam,)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      lm_scale:
        The scale of nn lm.
      LODR_lm_scale:
        The scale of the LODR_lm
      context_graph:
        A ContextGraph instance containing contextual phrases.
    Return:
      Returns the updated HypothesisList.
    """
    A = list(B)
    B = HypothesisList()
    for h in range(len(A)):
        hyp = A[h]
        for k in range(log_probs.size(0)):
            log_prob, index = log_probs[k], indexes[k]
            new_token = index.item()
            update_prefix = False
            new_hyp = hyp.clone()
            if new_token == blank_id:
                # Case 0: *a + ε => *a
                #         *aε + ε => *a
                # Prefix does not change, update log_prob of blank
                new_hyp.log_prob_non_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                new_hyp.log_prob_blank = hyp.log_prob + log_prob
                B.add(new_hyp)
            elif len(hyp.ys) > 0 and hyp.ys[-1] == new_token:
                # Case 1: *a + a => *a
                # Prefix does not change, update log_prob of non_blank
                new_hyp.log_prob_non_blank = hyp.log_prob_non_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                B.add(new_hyp)

                # Case 2: *aε + a => *aa
                # Prefix changes, update log_prob of blank
                new_hyp = hyp.clone()
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob_blank + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True
            else:
                # Case 3: *a + b => *ab, *aε + b => *ab
                # Prefix changes, update log_prob of non_blank
                # Caution: DO NOT use append, as clone is shallow copy
                new_hyp.ys = hyp.ys + [new_token]
                new_hyp.log_prob_non_blank = hyp.log_prob + log_prob
                new_hyp.log_prob_blank = torch.tensor(
                    [float("-inf")], dtype=torch.float32
                )
                update_prefix = True

            if update_prefix:
                lm_score = hyp.lm_score
                if hyp.lm_log_probs is not None:
                    lm_score = lm_score + hyp.lm_log_probs[new_token] * nnlm_scale
                    new_hyp.lm_log_probs = None

                if context_graph is not None and hyp.context_state is not None:
                    (
                        context_score,
                        new_context_state,
                        matched_state,
                    ) = context_graph.forward_one_step(hyp.context_state, new_token)
                    lm_score = lm_score + context_score
                    new_hyp.context_state = new_context_state

                if hyp.LODR_state is not None:
                    state_cost = hyp.LODR_state.forward_one_step(new_token)
                    # calculate the score of the latest token
                    current_ngram_score = state_cost.lm_score - hyp.LODR_state.lm_score
                    assert current_ngram_score <= 0.0, (
                        state_cost.lm_score,
                        hyp.LODR_state.lm_score,
                    )
                    lm_score = lm_score + LODR_lm_scale * current_ngram_score
                    new_hyp.LODR_state = state_cost

                new_hyp.lm_score = lm_score
                B.add(new_hyp)
    B = B.topk(beam)
    return B


def _sequence_worker(
    topk_values: torch.Tensor,
    topk_indexes: torch.Tensor,
    B: HypothesisList,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
) -> HypothesisList:
    """The worker to decode one sequence.
    Args:
      topk_values:
        topk log_probs of model output (i.e. the kept tokens of first pass pruning),
        the shape is (T, beam)
      topk_indexes:
        The indexes of the topk_values above, the shape is (T, beam)
      B:
        An instance of HypothesisList containing the kept hypothesis.
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
    Return:
      Returns the updated HypothesisList.
    """
    B.add(Hypothesis())
    for j in range(encoder_out_lens):
        log_probs, indexes = topk_values[j], topk_indexes[j]
        B = _step_worker(log_probs, indexes, B, beam, blank_id)
    return B


def ctc_prefix_beam_search(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    process_pool: Optional[Pool] = None,
    return_nbest: Optional[bool] = False,
) -> Union[List[List[int]], List[HypothesisList]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks".
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
      return_nbest:
        If true, return a list of HypothesisList, return a list of list of decoded token ids otherwise.
    """
    batch_size, num_frames, vocab_size = ctc_output.shape

    # TODO: using a larger beam for first pass pruning
    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()

    B = [HypothesisList() for _ in range(batch_size)]

    pool = Pool() if process_pool is None else process_pool
    arguments = []
    for i in range(batch_size):
        arguments.append(
            (
                topk_values[i],
                topk_indexes[i],
                B[i],
                encoder_out_lens[i].item(),
                beam,
                blank_id,
            )
        )
    async_results = pool.starmap_async(_sequence_worker, arguments)
    B = list(async_results.get())
    if process_pool is None:
        pool.close()
        pool.join()
    if return_nbest:
        return B
    else:
        best_hyps = [b.get_most_probable() for b in B]
        return [hyp.ys for hyp in best_hyps]


def ctc_prefix_beam_search_shallow_fussion(
    ctc_output: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    blank_id: int = 0,
    LODR_lm: Optional[NgramLm] = None,
    LODR_lm_scale: Optional[float] = 0,
    NNLM: Optional[LmScorer] = None,
    context_graph: Optional[ContextGraph] = None,
) -> List[List[int]]:
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks" and add
    nervous language model shallow fussion, it also supports contextual
    biasing with a given grammar.
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      LODR_lm:
        A low order n-gram LM, whose score will be subtracted during shallow fusion
      LODR_lm_scale:
        The scale of the LODR_lm
      LM:
        A neural net LM, e.g an RNNLM or transformer LM
      context_graph:
        A ContextGraph instance containing contextual phrases.
    Return:
      Returns a list of list of decoded token ids.
    """
    batch_size, num_frames, vocab_size = ctc_output.shape
    # TODO: using a larger beam for first pass pruning
    topk_values, topk_indexes = ctc_output.topk(beam)  # (B, T, beam)
    topk_values = topk_values.cpu()
    topk_indexes = topk_indexes.cpu()
    encoder_out_lens = encoder_out_lens.tolist()
    device = ctc_output.device

    nnlm_scale = 0
    init_scores = None
    init_states = None
    if NNLM is not None:
        nnlm_scale = NNLM.lm_scale
        sos_id = getattr(NNLM, "sos_id", 1)
        # get initial lm score and lm state by scoring the "sos" token
        sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
        lens = torch.tensor([1]).to(device)
        init_scores, init_states = NNLM.score_token(sos_token, lens)
        init_scores, init_states = init_scores.cpu(), (
            init_states[0].cpu(),
            init_states[1].cpu(),
        )

    B = [HypothesisList() for _ in range(batch_size)]
    for i in range(batch_size):
        B[i].add(
            Hypothesis(
                ys=[],
                log_prob_non_blank=torch.tensor([float("-inf")], dtype=torch.float32),
                log_prob_blank=torch.zeros(1, dtype=torch.float32),
                lm_score=torch.zeros(1, dtype=torch.float32),
                state=init_states,
                lm_log_probs=None if init_scores is None else init_scores.reshape(-1),
                LODR_state=None if LODR_lm is None else NgramLmStateCost(LODR_lm),
                context_state=None if context_graph is None else context_graph.root,
            )
        )
    for j in range(num_frames):
        for i in range(batch_size):
            if j < encoder_out_lens[i]:
                log_probs, indexes = topk_values[i][j], topk_indexes[i][j]
                B[i] = _step_worker(
                    log_probs=log_probs,
                    indexes=indexes,
                    B=B[i],
                    beam=beam,
                    blank_id=blank_id,
                    nnlm_scale=nnlm_scale,
                    LODR_lm_scale=LODR_lm_scale,
                    context_graph=context_graph,
                )
        if NNLM is None:
            continue
        # update lm_log_probs
        token_list = []  # a list of list
        hs = []
        cs = []
        indexes = []  # (batch_idx, key)
        for batch_idx, hyps in enumerate(B):
            for hyp in hyps:
                if hyp.lm_log_probs is None:  # those hyps that prefix changes
                    if NNLM.lm_type == "rnn":
                        token_list.append([hyp.ys[-1]])
                        # store the LSTM states
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        # for transformer LM
                        token_list.append([sos_id] + hyp.ys[:])
                    indexes.append((batch_idx, hyp.key))
        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if NNLM.lm_type == "rnn":
                tokens_to_score = (
                    torch.tensor(token_list).to(torch.int64).to(device).reshape(-1, 1)
                )
                hs = torch.cat(hs, dim=1).to(device)
                cs = torch.cat(cs, dim=1).to(device)
                state = (hs, cs)
            else:
                # for transformer LM
                tokens_list = [torch.tensor(tokens) for tokens in token_list]
                tokens_to_score = (
                    torch.nn.utils.rnn.pad_sequence(
                        tokens_list, batch_first=True, padding_value=0.0
                    )
                    .to(device)
                    .to(torch.int64)
                )
                state = None

            scores, lm_states = NNLM.score_token(tokens_to_score, x_lens, state)
            scores, lm_states = scores.cpu(), (lm_states[0].cpu(), lm_states[1].cpu())
            assert scores.size(0) == len(indexes), (scores.size(0), len(indexes))
            for i in range(scores.size(0)):
                batch_idx, key = indexes[i]
                B[batch_idx][key].lm_log_probs = scores[i]
                if NNLM.lm_type == "rnn":
                    state = (
                        lm_states[0][:, i, :].unsqueeze(1),
                        lm_states[1][:, i, :].unsqueeze(1),
                    )
                    B[batch_idx][key].state = state

    # finalize context_state, if the matched contexts do not reach final state
    # we need to add the score on the corresponding backoff arc
    if context_graph is not None:
        for hyps in B:
            for hyp in hyps:
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                hyp.lm_score += context_score
                hyp.context_state = new_context_state

    best_hyps = [b.get_most_probable() for b in B]
    return [hyp.ys for hyp in best_hyps]


def ctc_prefix_beam_search_attention_decoder_rescoring(
    ctc_output: torch.Tensor,
    attention_decoder: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 8,
    blank_id: int = 0,
    attention_scale: Optional[float] = None,
    process_pool: Optional[Pool] = None,
):
    """Implement prefix search decoding in "Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks" and add
    attention decoder rescoring.
    Args:
      ctc_output:
        The output of ctc head (log probability), the shape is (B, T, V)
      attention_decoder:
        The attention decoder.
      encoder_out:
        The output of encoder, the shape is (B, T, D)
      encoder_out_lens:
        The lengths (frames) of sequences after subsampling, the shape is (B,)
      beam:
        The number of hypothesis to be kept at each step.
      blank_id:
        The id of blank in the vocabulary.
      attention_scale:
        The scale of attention decoder score, if not provided it will search in
        a default list (see the code below).
      process_pool:
        The process pool for parallel decoding, if not provided, it will use all
        you cpu cores by default.
    """
    # List[HypothesisList]
    nbest = ctc_prefix_beam_search(
        ctc_output=ctc_output,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        blank_id=blank_id,
        return_nbest=True,
    )

    device = ctc_output.device

    hyp_shape = get_hyps_shape(nbest).to(device)
    hyp_to_utt_map = hyp_shape.row_ids(1).to(torch.long)
    # the shape of encoder_out is (N, T, C), so we use axis=0 here
    expanded_encoder_out = encoder_out.index_select(0, hyp_to_utt_map)
    expanded_encoder_out_lens = encoder_out_lens.index_select(0, hyp_to_utt_map)

    nbest = [list(x) for x in nbest]
    token_ids = []
    scores = []
    for hyps in nbest:
        for hyp in hyps:
            token_ids.append(hyp.ys)
            scores.append(hyp.log_prob.reshape(1))
    scores = torch.cat(scores).to(device)

    nll = attention_decoder.nll(
        encoder_out=expanded_encoder_out,
        encoder_out_lens=expanded_encoder_out_lens,
        token_ids=token_ids,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == len(token_ids)

    attention_scores = -nll.sum(dim=1)

    if attention_scale is None:
        attention_scale_list = [0.01, 0.05, 0.08]
        attention_scale_list += [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
        attention_scale_list += [2.1, 2.2, 2.3, 2.5, 3.0, 4.0, 5.0]
        attention_scale_list += [5.0, 6.0, 7.0, 8.0, 9.0]
    else:
        attention_scale_list = [attention_scale]

    ans = dict()

    start_indexes = hyp_shape.row_splits(1)[0:-1]
    for a_scale in attention_scale_list:
        tot_scores = scores + a_scale * attention_scores
        ragged_tot_scores = k2.RaggedTensor(hyp_shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        max_indexes = max_indexes - start_indexes
        max_indexes = max_indexes.cpu()
        best_path = [nbest[i][max_indexes[i]].ys for i in range(len(max_indexes))]
        key = f"attention_scale_{a_scale}"
        ans[key] = best_path
    return ans
