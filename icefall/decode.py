import logging
from typing import Dict, List

import k2
import torch


def _intersect_device(
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 50,
):
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
):
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
    Returns:
      A lattice containing the decoding result.
    """
    dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision_segments)

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
    lattice: k2.Fsa, num_paths: int, use_double_scores: bool = True
):
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
    Returns:
      An FsaVec containing linear FSAs.
    """
    # First, extract `num_paths` paths for each sequence.
    # path is a k2.RaggedInt with axes [seq][path][arc_pos]
    path = k2.random_paths(lattice, num_paths=num_paths, use_double_scores=True)

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


def compute_am_scores(
    lattice: k2.Fsa,
    word_fsa_with_epsilon_loops: k2.Fsa,
    path_to_seq_map: torch.Tensor,
) -> torch.Tensor:
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
      Return a 1-D torch.Tensor containing the AM scores of each path.
      `ans.numel() == word_fsas_with_epsilon_loops.shape[0]`
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

    am_path_lattice = _intersect_device(
        inv_lattice,
        word_fsa_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )

    am_path_lattice = k2.top_sort(k2.connect(am_path_lattice))

    # The `scores` of every arc consists of `am_scores` and `lm_scores`
    am_path_lattice.scores = am_path_lattice.scores - am_path_lattice.lm_scores

    am_scores = am_path_lattice.get_tot_scores(
        use_double_scores=True, log_semiring=False
    )

    return am_scores


def rescore_with_n_best_list(
    lattice: k2.Fsa, G: k2.Fsa, num_paths: int, lm_scale_list: List[float]
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

    # First, extract `num_paths` paths for each sequence.
    # path is a k2.RaggedInt with axes [seq][path][arc_pos]
    path = k2.random_paths(lattice, num_paths=num_paths, use_double_scores=True)

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

    am_scores = compute_am_scores(
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
    lattice: k2.Fsa, G_with_epsilon_loops: k2.Fsa, lm_scale_list: List[float]
) -> Dict[str, k2.Fsa]:
    """Use whole lattice to rescore.

    Args:
      lattice:
        An FsaVec It can be the return value of :func:`get_lattice`.
      G_with_epsilon_loops:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      lm_scale_list:
        A list containing lm_scale values.
    Returns:
      A dict of FsaVec, whose key is a lm_scale and the value represents the
      best decoding path for each sequence in the lattice.
    """
    assert len(lattice.shape) == 3
    assert hasattr(lattice, "lm_scores")
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lattice.device
    lattice.scores = lattice.scores - lattice.lm_scores
    # We will use lm_scores from G, so remove lats.lm_scores here
    del lattice.lm_scores
    assert hasattr(lattice, "lm_scores") is False

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
