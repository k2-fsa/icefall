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

from typing import Dict, List

import k2
import torch

from icefall.utils import get_texts


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
    k2.intersect_device.
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


# TODO(fangjun): Use Kangwei's C++ implementation that also
# supports List[List[int]]
def levenshtein_graph(symbol_ids: List[int]) -> k2.Fsa:
    """Construct a graph to compute Levenshtein distance.

    An example graph for `levenshtein_graph([1, 2, 3]` can be found
    at https://git.io/Ju7eW
    (Assuming the symbol table is 1<->a, 2<->b, 3<->c, blk<->0)

    Args:
      symbol_ids:
        A list of symbol IDs (excluding 0 and -1)
    """
    assert 0 not in symbol_ids
    assert -1 not in symbol_ids
    final_state = len(symbol_ids) + 1
    arcs = []
    for i in range(final_state - 1):
        arcs.append([i, i, 0, 0, -0.5])
        arcs.append([i, i + 1, 0, symbol_ids[i], -0.5])
        arcs.append([i, i + 1, symbol_ids[i], symbol_ids[i], 0])
    arcs.append([final_state - 1, final_state - 1, 0, 0, -0.5])
    arcs.append([final_state - 1, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    fsa = k2.arc_sort(fsa)
    return fsa


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
        lattice_score_scale: float = 0.5,
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
        lattice.scores *= lattice_score_scale
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
            word_seq = lattice.aux_labels.index(path)
            word_seq = word_seq.remove_axis(word_seq.num_axes - 2)

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
        return Nbest(fsa=fsa, shape=utt_to_path_shape)

    def intersect(self, lattice: k2.Fsa, use_double_scores=True) -> "Nbest":
        """Intersect this Nbest object with a lattice and get 1-best
        path from the resulting FsaVec.

        The purpose of this function is to attach scores to an Nbest.


        Args:
          lattice:
            An FsaVec with axes [utt][state][arc]. If it has `aux_labels`, then
            we assume its `labels` are token IDs and `aux_labels` are word IDs.
            If it has only `labels`, we assume it `labels` are word IDs.

          use_double_scores:
            True to use double precision when computing shortest path.
            False to use single precision.
        Returns:
          Return a new Nbest. This new Nbest shares the same shape with `self`,
          while its `fsa` is the 1-best path from intersecting `self.fsa` and
          `lattice`.
        """
        # Note: We view each linear FSA as a word sequence
        # and we use the passed lattice to give each word sequence a score.
        #
        # We are not viewing each linear FSAs as a token sequence.
        #
        # So we use k2.invert() here.

        # We use a word fsa to intersect with k2.invert(lattice)
        word_fsa = k2.invert(self.fsa)

        if hasattr(lattice, "aux_labels"):
            # delete token IDs as it is not needed
            del word_fsa.aux_labels

        word_fsa.scores.zero_()
        word_fsa_with_epsilon_loops = k2.remove_epsilon_and_add_self_loops(
            word_fsa
        )

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

        one_best = k2.shortest_path(
            path_lattice, use_double_scores=use_double_scores
        )

        one_best = k2.invert(one_best)
        # Now one_best has token IDs as labels and word IDs as aux_labels

        return Nbest(fsa=one_best, shape=self.shape)

    def compute_am_scores(self) -> k2.RaggedTensor:
        """Compute AM scores of each linear FSA (i.e., each path within
        an utterance).

        Hint:
          `self.fsa.scores` contains two parts: am scores and lm scores.

        Returns:
          Return a ragged tensor with 2 axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        saved_scores = self.fsa.scores

        # The `scores` of every arc consists of `am_scores` and `lm_scores`
        self.fsa.scores = self.fsa.scores - self.fsa.lm_scores

        am_scores = self.fsa.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        self.fsa.scores = saved_scores

        return k2.RaggedTensor(self.shape, am_scores)

    def tot_scores(self) -> k2.RaggedTensor:
        """Get total scores of the FSAs in this Nbest.

        Note:
          Since FSAs in Nbest are just linear FSAs, log-semirng and tropical
          semiring produce the same total scores.

        Returns:
          Return a ragged tensor with two axes [utt][path_scores].
          Its dtype is torch.float64.
        """
        # Use single precision since there are only additions.
        scores = self.fsa.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        return k2.RaggedTensor(self.shape, scores)

    def build_levenshtein_graphs(self) -> k2.Fsa:
        """Return an FsaVec with axes [utt][state][arc]."""
        word_ids = get_texts(self.fsa)
        word_levenshtein_graphs = [levenshtein_graph(ids) for ids in word_ids]
        return k2.Fsa.from_fsas(word_levenshtein_graphs)


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    lattice_score_scale: float = 1.0,
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.

    The basic idea is to first extra `num_paths` paths from the given lattice,
    build a word sequence from these paths, and compute the total scores
    of the word sequence in the tropical semiring. The one with the max score
    is used as the decoding output.

    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      **randomly**, not by ranking their scores.

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
      lattice_score_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
    Returns:
      An FsaVec containing linear FSAs. It axes are [utt][state][arc].
    """
    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
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
    lattice_score_scale: float = 0.5,
    oov: str = "<UNK>",
) -> Dict[str, List[List[int]]]:
    """Select the best hypothesis given a lattice and a reference transcript.

    The basic idea is to extract n paths from the given lattice, unique them,
    and select the one that has the minimum edit distance with the corresponding
    reference transcript as the decoding output.

    The decoding result returned from this function is the best result that
    we can obtain using n-best decoding with all kinds of rescoring techniques.

    This function is useful to tune the value of `lattice_score_scale`.

    Args:
      lattice:
        An FsaVec with axes [utt][state][arc].
        Note: We assume its aux_labels contain word IDs.
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
      lattice_score_scale:
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
        lattice_score_scale=lattice_score_scale,
    )

    hyps = nbest.build_levenshtein_graphs().to(device)

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
    levenshtein_graphs = [levenshtein_graph(ids) for ids in word_ids_list]
    refs = k2.Fsa.from_fsas(levenshtein_graphs).to(device)

    # Now compute the edit distance between hyps and refs
    hyps.rename_tensor_attribute_("aux_labels", "aux_labels2")
    edit_dist_lattice = k2.intersect_device(
        refs,
        hyps,
        b_to_a_map=nbest.shape.row_ids(1),
        sorted_match_a=True,
    )
    edit_dist_lattice = k2.remove_epsilon_self_loops(edit_dist_lattice)
    edit_dist_best_path = k2.shortest_path(
        edit_dist_lattice, use_double_scores=True
    ).invert_()
    edit_dist_best_path.rename_tensor_attribute_("aux_labels2", "aux_labels")

    tot_scores = edit_dist_best_path.get_tot_scores(
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
    lattice_score_scale: float = 1.0,
    use_double_scores: bool = True,
) -> Dict[str, k2.Fsa]:
    """Rescore a nbest list with an n-gram LM.
    The path with a maximum score is used as the decoding output.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert G.shape == (1, None, None)
    assert G.device == device
    assert hasattr(G, "aux_labels") is False

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        lattice_score_scale=lattice_score_scale,
    )
    # nbest.fsa.scores are all 0s at this point

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa has it scores set
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
