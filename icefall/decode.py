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

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import k2
import kaldialign
import torch
import torch.nn as nn

from .nbest import Nbest
from .utils import get_best_matching_stats

from .score_estimator import ScoreEstimator

def _get_random_paths(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    scale: float = 1.0,
):
    """
    Args:
      lattice:
        The decoding lattice, returned by :func:`get_lattice`.
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are remove dand only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
    Returns:
      Return a k2.RaggedInt with 3 axes [seq][path][arc_pos]
    """
    saved_scores = lattice.scores.clone()
    lattice.scores *= scale
    path = k2.random_paths(lattice, num_paths=num_paths, use_double_scores=True)
    lattice.scores = saved_scores
    return path


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

        fsas = k2.index(b_fsas, indexes)
        b_to_a = k2.index(b_to_a_map, indexes)
        path_lattice = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lattice)

    return k2.cat(ans)


def get_lattice(
    nnet_output: torch.Tensor,
    HLG: k2.Fsa,
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
        It is the output of a neural model of shape `[N, T, C]`.
      HLG:
        An Fsa, the decoding graph. See also `compile_HLG.py`.
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
      A lattice containing the decoding result.
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output, supervision_segments, allow_truncate=subsampling_factor - 1
    )

    lattice = k2.intersect_dense_pruned(
        HLG,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


def one_best_decoding(
    lattice: k2.Fsa, use_double_scores: bool = True
) -> k2.Fsa:
    """Get the best path from a lattice.

    Args:
      lattice:
        The decoding lattice returned by :func:`get_lattice`.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
    Return:
      An FsaVec containing linear paths.
    """
    best_path = k2.shortest_path(lattice, use_double_scores=use_double_scores)
    return best_path


def nbest_decoding(
    lattice: k2.Fsa,
    num_paths: int,
    use_double_scores: bool = True,
    scale: float = 1.0,
) -> k2.Fsa:
    """It implements something like CTC prefix beam search using n-best lists.

    The basic idea is to first extra n-best paths from the given lattice,
    build a word seqs from these paths, and compute the total scores
    of these sequences in the log-semiring. The one with the max score
    is used as the decoding output.

    Caution:
      Don't be confused by `best` in the name `n-best`. Paths are selected
      randomly, not by ranking their scores.

    Args:
      lattice:
        The decoding lattice, returned by :func:`get_lattice`.
      num_paths:
        It specifies the size `n` in n-best. Note: Paths are selected randomly
        and those containing identical word sequences are remove dand only one
        of them is kept.
      use_double_scores:
        True to use double precision floating point in the computation.
        False to use single precision.
      scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
    Returns:
      An FsaVec containing linear FSAs.
    """
    path = _get_random_paths(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        scale=scale,
    )

    # word_seq is a k2.RaggedInt sharing the same shape as `path`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seq = k2.index(lattice.aux_labels, path)
    # Note: the above operation supports also the case when
    # lattice.aux_labels is a ragged tensor. In that case,
    # `remove_axis=True` is used inside the pybind11 binding code,
    # so the resulting `word_seq` still has 3 axes, like `path`.
    # The 3 axes are [seq][path][word_id]

    # Remove 0 (epsilon) and -1 from word_seq
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)

    # Remove sequences with identical word sequences.
    #
    # k2.ragged.unique_sequences will reorder paths within a seq.
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.tot_size(1)
    unique_word_seq, _, new2old = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=False, need_new2old_indexes=True
    )
    # Note: unique_word_seq still has the same axes as word_seq

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seq.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path belongs
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seq has only two axes [path][word]
    unique_word_seq = k2.ragged.remove_axis(unique_word_seq, 0)

    # word_fsa is an FsaVec with axes [path][state][arc]
    word_fsa = k2.linear_fsa(unique_word_seq)

    # add epsilon self loops since we will use
    # k2.intersect_device, which treats epsilon as a normal symbol
    word_fsa_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsa)

    # lattice has token IDs as labels and word IDs as aux_labels.
    # inv_lattice has word IDs as labels and token IDs as aux_labels
    inv_lattice = k2.invert(lattice)
    inv_lattice = k2.arc_sort(inv_lattice)

    path_lattice = _intersect_device(
        inv_lattice,
        word_fsa_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )
    # path_lat has word IDs as labels and token IDs as aux_labels

    path_lattice = k2.top_sort(k2.connect(path_lattice))

    tot_scores = path_lattice.get_tot_scores(
        use_double_scores=use_double_scores, log_semiring=False
    )

    # RaggedFloat currently supports float32 only.
    # If Ragged<double> is wrapped, we can use k2.RaggedDouble here
    ragged_tot_scores = k2.RaggedFloat(
        seq_to_path_shape, tot_scores.to(torch.float32)
    )

    argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

    # Since we invoked `k2.ragged.unique_sequences`, which reorders
    # the index from `path`, we use `new2old` here to convert argmax_indexes
    # to the indexes into `path`.
    #
    # Use k2.index here since argmax_indexes' dtype is torch.int32
    best_path_indexes = k2.index(new2old, argmax_indexes)

    path_2axes = k2.ragged.remove_axis(path, 0)

    # best_path is a k2.RaggedInt with 2 axes [path][arc_pos]
    best_path = k2.index(path_2axes, best_path_indexes)

    # labels is a k2.RaggedInt with 2 axes [path][token_id]
    # Note that it contains -1s.
    labels = k2.index(lattice.labels.contiguous(), best_path)

    labels = k2.ragged.remove_values_eq(labels, -1)

    # lattice.aux_labels is a k2.RaggedInt tensor with 2 axes, so
    # aux_labels is also a k2.RaggedInt with 2 axes
    aux_labels = k2.index(lattice.aux_labels, best_path.values())

    best_path_fsa = k2.linear_fsa(labels)
    best_path_fsa.aux_labels = aux_labels
    return best_path_fsa


def compute_am_and_lm_scores(
    lattice: k2.Fsa,
    word_fsa_with_epsilon_loops: k2.Fsa,
    path_to_seq_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute AM scores of n-best lists (represented as word_fsas).

    Args:
      lattice:
        An FsaVec, e.g., the return value of :func:`get_lattice`
        It must have the attribute `lm_scores`.
      word_fsa_with_epsilon_loops:
        An FsaVec representing an n-best list. Note that it has been processed
        by `k2.add_epsilon_self_loops`.
      path_to_seq_map:
        A 1-D torch.Tensor with dtype torch.int32. path_to_seq_map[i] indicates
        which sequence the i-th Fsa in word_fsa_with_epsilon_loops belongs to.
        path_to_seq_map.numel() == word_fsas_with_epsilon_loops.arcs.dim0().
    Returns:
      Return a tuple containing two 1-D torch.Tensors: (am_scores, lm_scores).
      Each tensor's `numel()' equals to `word_fsas_with_epsilon_loops.shape[0]`
    """
    assert len(lattice.shape) == 3
    assert hasattr(lattice, "lm_scores")

    # k2.compose() currently does not support b_to_a_map. To void
    # replicating `lats`, we use k2.intersect_device here.
    #
    # lattice has token IDs as `labels` and word IDs as aux_labels, so we
    # need to invert it here.
    inv_lattice = k2.invert(lattice)

    # Now the `labels` of inv_lattice are word IDs (a 1-D torch.Tensor)
    # and its `aux_labels` are token IDs ( a k2.RaggedInt with 2 axes)

    # Remove its `aux_labels` since it is not needed in the
    # following computation
    del inv_lattice.aux_labels
    inv_lattice = k2.arc_sort(inv_lattice)

    path_lattice = _intersect_device(
        inv_lattice,
        word_fsa_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )

    path_lattice = k2.top_sort(k2.connect(path_lattice))

    # The `scores` of every arc consists of `am_scores` and `lm_scores`
    path_lattice.scores = path_lattice.scores - path_lattice.lm_scores

    am_scores = path_lattice.get_tot_scores(
        use_double_scores=True, log_semiring=False
    )

    path_lattice.scores = path_lattice.lm_scores

    lm_scores = path_lattice.get_tot_scores(
        use_double_scores=True, log_semiring=False
    )

    return am_scores.to(torch.float32), lm_scores.to(torch.float32)


def rescore_with_n_best_list(
    lattice: k2.Fsa,
    G: k2.Fsa,
    num_paths: int,
    lm_scale_list: List[float],
    scale: float = 1.0,
) -> Dict[str, k2.Fsa]:
    """Decode using n-best list with LM rescoring.

    `lattice` is a decoding lattice with 3 axes. This function first
    extracts `num_paths` paths from `lattice` for each sequence using
    `k2.random_paths`. The `am_scores` of these paths are computed.
    For each path, its `lm_scores` is computed using `G` (which is an LM).
    The final `tot_scores` is the sum of `am_scores` and `lm_scores`.
    The path with the largest `tot_scores` within a sequence is used
    as the decoding output.

    Args:
      lattice:
        An FsaVec. It can be the return value of :func:`get_lattice`.
      G:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      num_paths:
        It is the size `n` in `n-best` list.
      lm_scale_list:
        A list containing lm_scale values.
      scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
    Returns:
      A dict of FsaVec, whose key is an lm_scale and the value is the
      best decoding path for each sequence in the lattice.
    """
    device = lattice.device

    assert len(lattice.shape) == 3
    assert hasattr(lattice, "aux_labels")
    assert hasattr(lattice, "lm_scores")

    assert G.shape == (1, None, None)
    assert G.device == device
    assert hasattr(G, "aux_labels") is False

    path = _get_random_paths(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=True,
        scale=scale,
    )

    # word_seq is a k2.RaggedInt sharing the same shape as `path`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seq = k2.index(lattice.aux_labels, path)

    # Remove epsilons and -1 from word_seq
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)

    # Remove paths that has identical word sequences.
    #
    # unique_word_seq is still a k2.RaggedInt with 3 axes [seq][path][word]
    # except that there are no repeated paths with the same word_seq
    # within a sequence.
    #
    # num_repeats is also a k2.RaggedInt with 2 axes containing the
    # multiplicities of each path.
    # num_repeats.num_elements() == unique_word_seqs.tot_size(1)
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.tot_size(1)
    unique_word_seq, num_repeats, new2old = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=True, need_new2old_indexes=True
    )

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seq.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seq has only two axes [path][word]
    unique_word_seq = k2.ragged.remove_axis(unique_word_seq, 0)

    # word_fsa is an FsaVec with axes [path][state][arc]
    word_fsa = k2.linear_fsa(unique_word_seq)

    word_fsa_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsa)

    am_scores, _ = compute_am_and_lm_scores(
        lattice, word_fsa_with_epsilon_loops, path_to_seq_map
    )

    # Now compute lm_scores
    b_to_a_map = torch.zeros_like(path_to_seq_map)
    lm_path_lattice = _intersect_device(
        G,
        word_fsa_with_epsilon_loops,
        b_to_a_map=b_to_a_map,
        sorted_match_a=True,
    )
    lm_path_lattice = k2.top_sort(k2.connect(lm_path_lattice))
    lm_scores = lm_path_lattice.get_tot_scores(
        use_double_scores=True, log_semiring=False
    )

    path_2axes = k2.ragged.remove_axis(path, 0)

    ans = dict()
    for lm_scale in lm_scale_list:
        tot_scores = am_scores / lm_scale + lm_scores

        # Remember that we used `k2.ragged.unique_sequences` to remove repeated
        # paths to avoid redundant computation in `k2.intersect_device`.
        # Now we use `num_repeats` to correct the scores for each path.
        #
        # NOTE(fangjun): It is commented out as it leads to a worse WER
        # tot_scores = tot_scores * num_repeats.values()

        ragged_tot_scores = k2.RaggedFloat(
            seq_to_path_shape, tot_scores.to(torch.float32)
        )
        argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

        # Use k2.index here since argmax_indexes' dtype is torch.int32
        best_path_indexes = k2.index(new2old, argmax_indexes)

        # best_path is a k2.RaggedInt with 2 axes [path][arc_pos]
        best_path = k2.index(path_2axes, best_path_indexes)

        # labels is a k2.RaggedInt with 2 axes [path][phone_id]
        # Note that it contains -1s.
        labels = k2.index(lattice.labels.contiguous(), best_path)

        labels = k2.ragged.remove_values_eq(labels, -1)

        # lattice.aux_labels is a k2.RaggedInt tensor with 2 axes, so
        # aux_labels is also a k2.RaggedInt with 2 axes
        aux_labels = k2.index(lattice.aux_labels, best_path.values())

        best_path_fsa = k2.linear_fsa(labels)
        best_path_fsa.aux_labels = aux_labels

        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path_fsa

    return ans


def rescore_with_whole_lattice(
    lattice: k2.Fsa,
    G_with_epsilon_loops: k2.Fsa,
    lm_scale_list: Optional[List[float]] = None,
) -> Union[k2.Fsa, Dict[str, k2.Fsa]]:
    """Use whole lattice to rescore.

    Args:
      lattice:
        An FsaVec It can be the return value of :func:`get_lattice`.
      G_with_epsilon_loops:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      lm_scale_list:
        A list containing lm_scale values or None.
    Returns:
      If lm_scale_list is not None, return a dict of FsaVec, whose key
      is a lm_scale and the value represents the best decoding path for
      each sequence in the lattice.
      If lm_scale_list is not None, return a lattice that is rescored
      with the given LM.
    """
    assert len(lattice.shape) == 3
    assert hasattr(lattice, "lm_scores")
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lattice.device
    lattice.scores = lattice.scores - lattice.lm_scores
    # We will use lm_scores from G, so remove lats.lm_scores here
    del lattice.lm_scores
    assert hasattr(lattice, "lm_scores") is False

    assert hasattr(G_with_epsilon_loops, "lm_scores")

    # Now, lattice.scores contains only am_scores

    # inv_lattice has word IDs as labels.
    # Its aux_labels are token IDs, which is a ragged tensor k2.RaggedInt
    inv_lattice = k2.invert(lattice)
    num_seqs = lattice.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    while True:
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
            logging.info(
                f"num_arcs before pruning: {inv_lattice.arcs.num_elements()}"
            )

            # NOTE(fangjun): The choice of the threshold 1e-7 is arbitrary here
            # to avoid OOM. We may need to fine tune it.
            inv_lattice = k2.prune_on_arc_post(inv_lattice, 1e-7, True)
            logging.info(
                f"num_arcs after pruning: {inv_lattice.arcs.num_elements()}"
            )

    # lat has token IDs as labels
    # and word IDs as aux_labels.
    lat = k2.invert(rescoring_lattice)

    if lm_scale_list is None:
        return lat

    ans = dict()
    #
    # The following implements
    # scores = (scores - lm_scores)/lm_scale + lm_scores
    #        = scores/lm_scale + lm_scores*(1 - 1/lm_scale)
    #
    saved_am_scores = lat.scores - lat.lm_scores
    for lm_scale in lm_scale_list:
        am_scores = saved_am_scores / lm_scale
        lat.scores = am_scores + lat.lm_scores

        best_path = k2.shortest_path(lat, use_double_scores=True)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_path
    return ans


def nbest_oracle(
    lattice: k2.Fsa,
    num_paths: int,
    ref_texts: List[str],
    word_table: k2.SymbolTable,
    scale: float = 1.0,
) -> Dict[str, List[List[int]]]:
    """Select the best hypothesis given a lattice and a reference transcript.

    The basic idea is to extract n paths from the given lattice, unique them,
    and select the one that has the minimum edit distance with the corresponding
    reference transcript as the decoding output.

    The decoding result returned from this function is the best result that
    we can obtain using n-best decoding with all kinds of rescoring techniques.

    Args:
      lattice:
        An FsaVec. It can be the return value of :func:`get_lattice`.
        Note: We assume its aux_labels contain word IDs.
      num_paths:
        The size of `n` in n-best.
      ref_texts:
        A list of reference transcript. Each entry contains space(s)
        separated words
      word_table:
        It is the word symbol table.
      scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
    Return:
      Return a dict. Its key contains the information about the parameters
      when calling this function, while its value contains the decoding output.
      `len(ans_dict) == len(ref_texts)`
    """
    path = _get_random_paths(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=True,
        scale=scale,
    )

    word_seq = k2.index(lattice.aux_labels, path)
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)
    unique_word_seq, _, _ = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=False, need_new2old_indexes=False
    )
    unique_word_ids = k2.ragged.to_list(unique_word_seq)
    assert len(unique_word_ids) == len(ref_texts)
    # unique_word_ids[i] contains all hypotheses of the i-th utterance

    results = []
    for hyps, ref in zip(unique_word_ids, ref_texts):
        # Note hyps is a list-of-list ints
        # Each sublist contains a hypothesis
        ref_words = ref.strip().split()
        # CAUTION: We don't convert ref_words to ref_words_ids
        # since there may exist OOV words in ref_words
        best_hyp_words = None
        min_error = float("inf")
        for hyp_words in hyps:
            hyp_words = [word_table[i] for i in hyp_words]
            this_error = kaldialign.edit_distance(ref_words, hyp_words)["total"]
            if this_error < min_error:
                min_error = this_error
                best_hyp_words = hyp_words
        results.append(best_hyp_words)

    return {f"nbest_{num_paths}_scale_{scale}_oracle": results}


def rescore_with_attention_decoder(
    lattice: k2.Fsa,
    num_paths: int,
    model: nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: Optional[torch.Tensor],
    sos_id: int,
    eos_id: int,
    scale: float = 1.0,
    ngram_lm_scale: Optional[float] = None,
    attention_scale: Optional[float] = None,
) -> Dict[str, k2.Fsa]:
    """This function extracts n paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest
    score is used as the decoding output.

    Args:
      lattice:
        An FsaVec. It can be the return value of :func:`get_lattice`.
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `[T, N, C]`.
      memory_key_padding_mask:
        The padding mask for memory with shape [N, T].
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
      scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      ngram_lm_scale:
        Optional. It specifies the scale for n-gram LM scores.
      attention_scale:
        Optional. It specifies the scale for attention decoder scores.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each sequence in the lattice.
    """
    # First, extract `num_paths` paths for each sequence.
    # path is a k2.RaggedInt with axes [seq][path][arc_pos]
    path = _get_random_paths(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=True,
        scale=scale,
    )

    # word_seq is a k2.RaggedInt sharing the same shape as `path`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seq = k2.index(lattice.aux_labels, path)

    # Remove epsilons and -1 from word_seq
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)

    # Remove paths that has identical word sequences.
    #
    # unique_word_seq is still a k2.RaggedInt with 3 axes [seq][path][word]
    # except that there are no repeated paths with the same word_seq
    # within a sequence.
    #
    # num_repeats is also a k2.RaggedInt with 2 axes containing the
    # multiplicities of each path.
    # num_repeats.num_elements() == unique_word_seqs.tot_size(1)
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seq.tot_size(1)
    unique_word_seq, num_repeats, new2old = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=True, need_new2old_indexes=True
    )

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seq.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seq has only two axes [path][word]
    unique_word_seq = k2.ragged.remove_axis(unique_word_seq, 0)

    # word_fsa is an FsaVec with axes [path][state][arc]
    word_fsa = k2.linear_fsa(unique_word_seq)

    word_fsa_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsa)

    am_scores, ngram_lm_scores = compute_am_and_lm_scores(
        lattice, word_fsa_with_epsilon_loops, path_to_seq_map
    )
    # Now we use the attention decoder to compute another
    # score: attention_scores.
    #
    # To do that, we have to get the input and output for the attention
    # decoder.

    # CAUTION: The "tokens" attribute is set in the file
    # local/compile_hlg.py
    token_seq = k2.index(lattice.tokens, path)

    # Remove epsilons and -1 from token_seq
    token_seq = k2.ragged.remove_values_leq(token_seq, 0)

    # Remove the seq axis.
    token_seq = k2.ragged.remove_axis(token_seq, 0)

    token_seq, _ = k2.ragged.index(
        token_seq, indexes=new2old, axis=0, need_value_indexes=False
    )

    # Now word in unique_word_seq has its corresponding token IDs.
    token_ids = k2.ragged.to_list(token_seq)

    num_word_seqs = new2old.numel()

    path_to_seq_map_long = path_to_seq_map.to(torch.long)
    expanded_memory = memory.index_select(1, path_to_seq_map_long)

    if memory_key_padding_mask is not None:
        expanded_memory_key_padding_mask = memory_key_padding_mask.index_select(
            0, path_to_seq_map_long
        )
    else:
        expanded_memory_key_padding_mask = None

    nll = model.decoder_nll(
        memory=expanded_memory,
        memory_key_padding_mask=expanded_memory_key_padding_mask,
        token_ids=token_ids,
        sos_id=sos_id,
        eos_id=eos_id,
    )
    assert nll.ndim == 2
    assert nll.shape[0] == num_word_seqs

    attention_scores = -nll.sum(dim=1)
    assert attention_scores.ndim == 1
    assert attention_scores.numel() == num_word_seqs

    if ngram_lm_scale is None:
        ngram_lm_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        ngram_lm_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
    else:
        ngram_lm_scale_list = [ngram_lm_scale]

    if attention_scale is None:
        attention_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
        attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]
    else:
        attention_scale_list = [attention_scale]

    path_2axes = k2.ragged.remove_axis(path, 0)

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            tot_scores = (
                am_scores
                + n_scale * ngram_lm_scores
                + a_scale * attention_scores
            )
            ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape, tot_scores)
            argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

            best_path_indexes = k2.index(new2old, argmax_indexes)

            # best_path is a k2.RaggedInt with 2 axes [path][arc_pos]
            best_path = k2.index(path_2axes, best_path_indexes)

            # labels is a k2.RaggedInt with 2 axes [path][token_id]
            # Note that it contains -1s.
            labels = k2.index(lattice.labels.contiguous(), best_path)

            labels = k2.ragged.remove_values_eq(labels, -1)

            # lattice.aux_labels is a k2.RaggedInt tensor with 2 axes, so
            # aux_labels is also a k2.RaggedInt with 2 axes
            aux_labels = k2.index(lattice.aux_labels, best_path.values())

            best_path_fsa = k2.linear_fsa(labels)
            best_path_fsa.aux_labels = aux_labels

            key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}"
            ans[key] = best_path_fsa
    return ans


def rescore_nbest_with_attention_decoder(
    nbest: Nbest,
    model: nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    sos_id: int,
    eos_id: int,
) -> Nbest:
    """This function rescores an nbest list with an attention decoder. The paths
    with rescored scores are returned as a new nbest.

    Args:
      nbest:
        An Nbest, the nbest path of given sequences.
        It can be the return value of :func:`generate_nbest_list`.
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `[T, N, C]`.
      memory_key_padding_mask:
        The padding mask for memory with shape [N, T].
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
    Returns:
      A Nbest with all of the scores on fsa arcs updated with attention scores.
    """
    num_paths = nbest.shape.num_elements()
    # token shape [utt][path][state][arc]
    token_shape = k2.ragged.compose_ragged_shapes(nbest.shape, nbest.fsa.arcs.shape())

    token_seq = k2.RaggedInt(token_shape, nbest.fsa.labels.contiguous())

    # Remove -1 from token_seq, there is no epsilon tokens in token_seq, we
    # removed it when generating nbest list
    token_seq = k2.ragged.remove_values_leq(token_seq, -1)
    # token seq shape [utt][path][token]
    token_seq = k2.ragged.remove_axis(token_seq, 2)
    # token seq shape [utt][token]
    token_seq = k2.ragged.remove_axis(token_seq, 0)

    token_ids = k2.ragged.to_list(token_seq)

    path_to_seq_map_long = nbest.shape.row_ids(1).to(torch.long)
    expanded_memory = memory.index_select(1, path_to_seq_map_long)

    expanded_memory_key_padding_mask = memory_key_padding_mask.index_select(
        0, path_to_seq_map_long
    )

    # TODO: pass the sos_token_id and eos_token_id via function arguments
    nll = model.decoder_nll(
        memory=expanded_memory,
        memory_key_padding_mask=expanded_memory_key_padding_mask,
        token_ids=token_ids,
        sos_id=sos_id,
        eos_id=eos_id,
    )

    assert nll.ndim == 2
    assert nll.shape[0] == num_paths

    attention_scores = torch.zeros(
        nbest.fsa.scores.size()[0],
        dtype=torch.float32,
        device=nbest.fsa.device
    )

    start_index = 0
    for i in range(num_paths):
        # Plus 1 to fill the score of final arc
        tokens_num = 0 if len(token_ids[i]) == 0 else len(token_ids[i]) + 1
        attention_scores[start_index: start_index + tokens_num] =\
            nll[i][0: tokens_num]
        start_index += tokens_num

    fsas = nbest.fsa.clone()
    fsas.scores = attention_scores
    return Nbest(fsas, nbest.shape)


def rescore_with_attention_decoder_v2(
    lattice: k2.Fsa,
    batch_idx: int,
    dump_best_matching_feature: bool,
    num_paths: int,
    top_k: int,
    model: nn.Module,
    memory: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
    rescore_est_model: nn.Module,
    sos_id: int,
    eos_id: int,
) -> Union[torch.Tensor, Dict[str, k2.Fsa]]:
    """This function extracts n paths from the given lattice and uses
    an attention decoder to rescore them. The path with the highest
    score is used as the decoding output.

    Args:
      lattice:
        An FsaVec. It can be the return value of :func:`get_lattice`.
      batch_idx:
        The batch index currently processed.
      dump_best_matching_feature:
        Whether to dump best matching feature, only for preparing training
        data for attention-decoder-v2
      num_paths:
        Number of paths to extract from the given lattice for rescoring.
      model:
        A transformer model. See the class "Transformer" in
        conformer_ctc/transformer.py for its interface.
      memory:
        The encoder memory of the given model. It is the output of
        the last torch.nn.TransformerEncoder layer in the given model.
        Its shape is `[T, N, C]`.
      memory_key_padding_mask:
        The padding mask for memory with shape [N, T].
      rescore_est_model:
        The model to estimate rescore mean and variance, only for attention-decoder-v2
      sos_id:
        The token ID for SOS.
      eos_id:
        The token ID for EOS.
    Returns:
      A dict of FsaVec, whose key contains a string
      ngram_lm_scale_attention_scale and the value is the
      best decoding path for each sequence in the lattice.
    """
    nbest = generate_nbest_list(lattice, num_paths)

    if dump_best_matching_feature:
        if nbest.fsa.arcs.dim0() <= 2 * top_k or nbest.fsa.arcs.num_elements() == 0:
            return torch.empty(0)
        nbest_k, nbest_q = nbest.split(k=top_k, sort=False)

        rescored_nbest_k = rescore_nbest_with_attention_decoder(
            nbest=nbest_k,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id
        )
        stats_tensor = get_best_matching_stats(
            rescored_nbest_k,
            nbest_q,
            max_order=5
        )
        rescored_nbest_q = rescore_nbest_with_attention_decoder(
            nbest=nbest_q,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id
        )
        merge_tensor = torch.cat(
            (stats_tensor, rescored_nbest_q.fsa.scores.clone().view(-1, 1)),
            dim=1
        )
        return merge_tensor

    if nbest.fsa.arcs.dim0() >= 2 * top_k and nbest.fsa.arcs.num_elements() != 0:
        nbest_topk, nbest_remain = nbest.split(k=top_k)

        am_scores = nbest_topk.fsa.scores - nbest_topk.fsa.lm_scores
        lm_scores = nbest_topk.fsa.lm_scores

        rescored_nbest_topk = rescore_nbest_with_attention_decoder(
            nbest=nbest_topk,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        stats_tensor = get_best_matching_stats(
            rescored_nbest_topk,
            nbest_remain,
            max_order=5
        )

        # run rescore estimation model to get the mean and var of each token
        mean, var = rescore_est_model(stats_tensor)

        # mean_shape [utt][path][state][arcs]
        mean_shape = k2.ragged.compose_ragged_shapes(
            nbest_remain.shape, nbest_remain.fsa.arcs.shape())
        # mean_shape [utt][path][arcs]
        mean_shape = k2.ragged.remove_axis(mean_shape, 2)
        ragged_mean = k2.RaggedFloat(mean_shape, mean.contiguous())
        # path mean shape [utt][path]
        path_mean = k2.ragged.sum_per_sublist(ragged_mean)

        # var_shape  [utt][path][state][arcs]
        var_shape = k2.ragged.compose_ragged_shapes(
            nbest_remain.shape, nbest_remain.fsa.arcs.shape())
        # var_shape  [utt][path][arcs]
        var_shape = k2.ragged.remove_axis(var_shape, 2)
        ragged_var = k2.RaggedFloat(var_shape, var.contiguous())
        # path var shape [utt][path]
        path_var = k2.ragged.sum_per_sublist(ragged_var)

        # tot_scores() shape [utt][path]
        # path_score with elements numbers equals numbers of paths
        # !!! Note: This is right only when utt equals to 1
        path_scores = nbest_remain.total_scores().values()
        best_score = torch.max(rescored_nbest_topk.total_scores().values())
        est_scores = 1 - 1/2 * (
            1 + torch.erf(
                (best_score - path_mean) / torch.sqrt(2 * path_var)
            )
        )
        est_scores = k2.RaggedFloat(nbest_remain.shape, est_scores)

        # calculate nbest_remain estimated score and select topk
        nbest_remain_topk = nbest_remain.top_k(k=top_k, scores=est_scores)
        remain_am_scores = nbest_remain_topk.fsa.scores - nbest_remain_topk.fsa.lm_scores
        remain_lm_scores = nbest_remain_topk.fsa.lm_scores
        rescored_nbest_remain_topk = rescore_nbest_with_attention_decoder(
            nbest=nbest_remain_topk,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id,
        )

        # !!! Note: This is right only when utt equals to 1
        merge_fsa = k2.cat([rescored_nbest_topk.fsa, rescored_nbest_remain_topk.fsa])
        merge_row_ids = torch.zeros(
            merge_fsa.arcs.dim0(),
            dtype=torch.int32,
            device=merge_fsa.device
        )
        rescore_nbest = Nbest(
            merge_fsa, k2.ragged.create_ragged_shape2(row_ids=merge_row_ids)
        )

        attention_scores = rescore_nbest.fsa.scores
        am_scores = torch.cat((am_scores, remain_am_scores))
        lm_scores = torch.cat((lm_scores, remain_lm_scores))
    else:
        am_scores = nbest.fsa.scores - nbest.fsa.lm_scores
        lm_scores = nbest.fsa.lm_scores
        rescore_nbest = rescore_nbest_with_attention_decoder(
            nbest=nbest,
            model=model,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            sos_id=sos_id,
            eos_id=eos_id
        )
        attention_scores = rescore_nbest.fsa.scores

    ngram_lm_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
    ngram_lm_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]

    attention_scale_list = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
    attention_scale_list += [1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.0]

    ans = dict()
    for n_scale in ngram_lm_scale_list:
        for a_scale in attention_scale_list:
            tot_scores = (
                am_scores
                + n_scale * lm_scores
                + a_scale * attention_scores
            )
            rescore_nbest.fsa.scores = tot_scores
            # ragged tot scores shape [utt][path]
            ragged_tot_scores = rescore_nbest.total_scores()

            argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

            best_fsas = k2.index_fsa(rescore_nbest.fsa, argmax_indexes)

            key = f"ngram_lm_scale_{n_scale}_attention_scale_{a_scale}"
            ans[key] = best_fsas
    return ans


def generate_nbest_list(
    lats: k2.Fsa,
    num_paths: int,
    aux_labels: bool = False
) -> Nbest:
    '''Generate an n-best list from a lattice.

    Args:
      lats:
        The decoding lattice from the first pass after LM rescoring.
        lats is an FsaVec. It can be the return value of
        :func:`rescore_with_whole_lattice`
      num_paths:
        Size of n for n-best list. CAUTION: After removing paths
        that represent the same word sequences, the number of paths
        in different sequences may not be equal.
    Return:
      Return an Nbest object.
    '''
    # First, extract `num_paths` paths for each sequence.
    # path is a k2.RaggedInt with axes [seq][path][arc_pos]
    path = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # word_seq is a k2.RaggedInt sharing the same shape as `path`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.
    word_seq = k2.index(lats.aux_labels, path)

    # Remove epsilons and -1 from word_seq
    word_seq = k2.ragged.remove_values_leq(word_seq, 0)

    # Remove paths that has identical word sequences.
    #
    # unique_word_seq is still a k2.RaggedInt with 3 axes [seq][path][word]
    # except that there are no repeated paths with the same word_seq
    # within a sequence.
    #
    # num_repeats is also a k2.RaggedInt with 2 axes containing the
    # multiplicities of each path.
    # num_repeats.num_elements() == unique_word_seqs.num_elements()
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.tot_size(1)
    unique_word_seq, num_repeats, new2old = k2.ragged.unique_sequences(
        word_seq, need_num_repeats=True, need_new2old_indexes=True
    )

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seq.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seq has only two axes [path][word]
    unique_word_seq = k2.ragged.remove_axis(unique_word_seq, 0)

    # word_fsa is an FsaVec with axes [path][state][arc]
    word_fsa = k2.linear_fsa(unique_word_seq)

    word_fsa_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsa)

    # k2.compose() currently does not support b_to_a_map. To void
    # replicating `lats`, we use k2.intersect_device here.
    #
    # lattice has token IDs as `labels` and word IDs as aux_labels, so we
    # need to invert it here.
    inv_lattice = k2.invert(lats)

    # Now the `labels` of inv_lattice are word IDs (a 1-D torch.Tensor)
    # and its `aux_labels` are token IDs ( a k2.RaggedInt with 2 axes)

    # Remove its `aux_labels` since it is not needed in the
    # following computation
    # del inv_lattice.aux_labels
    inv_lattice = k2.arc_sort(inv_lattice)

    path_lattice = _intersect_device(
        inv_lattice,
        word_fsa_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )

    # path_lattice now has token IDs as `labels` and word IDS as aux_labels.
    path_lattice = k2.invert(path_lattice)

    path_lattice = k2.top_sort(k2.connect(path_lattice))

    # replace labels with tokens to remove repeat token IDs.
    path_lattice.labels = path_lattice.tokens

    n_best = k2.shortest_path(path_lattice, use_double_scores=True)

    n_best = k2.remove_epsilon(n_best)

    n_best = k2.top_sort(k2.connect(n_best))

    # now we have nbest lists with am_scores and lm_scores
    return Nbest(fsa=n_best, shape=seq_to_path_shape)

