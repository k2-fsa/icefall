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

"""
This file contains rescoring code for NN LMs, e.g., conformer LM.

Here are the ideas about preparing the inputs for the conformer LM model
from an Nbest object.

Given an Nbest object `nbest`, we have:
    - nbest.fsa
    - nbest.shape, whose axes are [utt][path]

We can get `tokens` from nbest.fsa. The resulting `tokens` will have
2 axes [path][token]. Note, we should remove 0s from `tokens`.

We can generate the following inputs for the conformer LM model from `tokens`:
    - masked_src
    - src
    - tgt
by using `k2.levenshtein_alignment`.
"""

from typing import Tuple

import k2
import torch

from icefall.decode import Nbest


def concat(
    ragged: k2.RaggedTensor, value: int, direction: str
) -> k2.RaggedTensor:
    """Prepend a value to the beginning of each sublist or append a value.
    to the end of each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      value:
        The value to prepend or append.
      direction:
        It can be either "left" or "right". If it is "left", we
        prepend the value to the beginning of each sublist;
        if it is "right", we append the value to the end of each
        sublist.

    Returns:
      Return a new ragged tensor, whose sublists either start with
      or end with the given value.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> concat(a, value=0, direction="left")
    [ [ 0 1 3 ] [ 0 5 ] ]
    >>> concat(a, value=0, direction="right")
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    dtype = ragged.dtype
    device = ragged.device

    assert ragged.num_axes == 2, f"num_axes: {ragged.num_axes}"
    pad_values = torch.full(
        size=(ragged.tot_size(0),),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad_shape = k2.ragged.regular_ragged_shape(ragged.tot_size(0), 1).to(device)
    pad = k2.RaggedTensor(pad_shape, pad_values)

    if direction == "left":
        ans = k2.ragged.cat([pad, ragged], axis=1)
    elif direction == "right":
        ans = k2.ragged.cat([ragged, pad], axis=1)
    else:
        raise ValueError(
            f'Unsupported direction: {direction}. " \
            "Expect either "left" or "right"'
        )
    return ans


def add_bos(ragged: k2.RaggedTensor, bos_id: int) -> k2.RaggedTensor:
    """Add BOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      bos_id:
        The ID of the BOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist starts with BOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_bos(a, bos_id=0)
    [ [ 0 1 3 ] [ 0 5 ] ]

    """
    return concat(ragged, bos_id, direction="left")


def add_eos(ragged: k2.RaggedTensor, eos_id: int) -> k2.RaggedTensor:
    """Add EOS to each sublist.

    Args:
      ragged:
        A ragged tensor with two axes.
      bos_id:
        The ID of the EOS symbol.

    Returns:
      Return a new ragged tensor, where each sublist ends with EOS.

    >>> a = k2.RaggedTensor([[1, 3], [5]])
    >>> a
    [ [ 1 3 ] [ 5 ] ]
    >>> add_eos(a, eos_id=0)
    [ [ 1 3 0 ] [ 5 0 ] ]

    """
    return concat(ragged, eos_id, direction="right")


def make_hyp_to_ref_map(row_splits: torch.Tensor):
    """
    TODO: Add documentation.

    >>> row_splits = torch.tensor([0, 3, 5], dtype=torch.int32)
    >>> make_hyp_to_ref_map(row_splits)
    tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], dtype=torch.int32)

    """
    device = row_splits.device
    sizes = (row_splits[1:] - row_splits[:-1]).tolist()
    offsets = row_splits[:-1]

    map_tensor_list = []
    for size, offset in zip(sizes, offsets):
        # Explanation of the following operations
        # assume size is 3, offset is 2
        # torch.arange() + offset is [2, 3, 4]
        # expand() is [[2, 3, 4], [2, 3, 4], [2, 3, 4]]
        # t() is [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
        # reshape() is [2, 2, 2, 3, 3, 3, 4, 4, 4]
        map_tensor = (
            (torch.arange(size, dtype=torch.int32, device=device) + offset)
            .expand(size, size)
            .t()
            .reshape(-1)
        )
        map_tensor_list.append(map_tensor)

    return torch.cat(map_tensor_list)


def make_repeat_map(row_splits: torch.Tensor):
    """
    TODO: Add documentation.

    >>> row_splits = torch.tensor([0, 3, 5], dtype=torch.int32)
    >>> make_repeat_map(row_splits)
    tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 3, 4], dtype=torch.int32)

    """
    device = row_splits.device
    sizes = (row_splits[1:] - row_splits[:-1]).tolist()
    offsets = row_splits[:-1]

    map_tensor_list = []
    for size, offset in zip(sizes, offsets):
        # Explanation of the following operations
        # assume size is 3, offset is 2
        # torch.arange() + offset is [2, 3, 4]
        # expand() is [[2, 3, 4], [2, 3, 4], [2, 3, 4]]
        # reshape() is [2, 3, 4, 2, 3, 4, 2, 3, 4]
        map_tensor = (
            (torch.arange(size, dtype=torch.int32, device=device) + offset)
            .expand(size, size)
            .reshape(-1)
        )
        map_tensor_list.append(map_tensor)

    return torch.cat(map_tensor_list)


def make_repeat(tokens: k2.RaggedTensor) -> k2.RaggedTensor:
    """Repeat the number of paths of an utterance to the number that
    equals to the number of paths in the utterance.

    For instance, if an utterance contains 3 paths: [path1 path2 path3],
    after repeating, this utterance will contain 9 paths:
    [path1 path2 path3] [path1 path2 path3] [path1 path2 path3]

    >>> tokens = k2.RaggedTensor([ [[1, 2, 3], [4, 5], [9]], [[5, 8], [10, 1]] ])
    >>> tokens
    [ [ [ 1 2 3 ] [ 4 5 ] [ 9 ] ] [ [ 5 8 ] [ 10 1 ] ] ]
    >>> make_repeat(tokens)
    [ [ [ 1 2 3 ] [ 4 5 ] [ 9 ] [ 1 2 3 ] [ 4 5 ] [ 9 ] [ 1 2 3 ] [ 4 5 ] [ 9 ] ] [ [ 5 8 ] [ 10 1 ] [ 5 8 ] [ 10 1 ] ] ]

    TODO: Add documentation.

    """
    assert tokens.num_axes == 3, f"num_axes: {tokens.num_axes}"
    if True:
        indexes = make_repeat_map(tokens.shape.row_splits(1))
        return tokens.index(axis=1, indexes=indexes)[0]
    else:
        # This branch produces the same result as the above branch.
        # It's more readable. Will remove it later.
        repeated = []
        for p in tokens.tolist():
            repeated.append(p * len(p))
        return k2.RaggedTensor(repeated).to(tokens.device)


def compute_alignments(
    tokens: k2.RaggedTensor,
    shape: k2.RaggedShape,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TODO: Add documentation.

    Args:
      tokens:
        A ragged tensor with two axes: [path][token].
      shape:
        A ragged shape with two axes: [utt][path]
    """
    assert tokens.tot_size(0) == shape.tot_size(1)
    device = tokens.device
    utt_path_shape = shape.compose(tokens.shape)
    utt_path_token = k2.RaggedTensor(utt_path_shape, tokens.values)
    utt_path_token_repeated = make_repeat(utt_path_token)
    path_token_repeated = utt_path_token_repeated.remove_axis(0)

    refs = k2.levenshtein_graph(tokens, device=device)
    hyps = k2.levenshtein_graph(path_token_repeated, device=device)

    hyp_to_ref_map = make_hyp_to_ref_map(utt_path_shape.row_splits(1))
    alignment = k2.levenshtein_alignment(
        refs=refs, hyps=hyps, hyp_to_ref_map=hyp_to_ref_map
    )
    return alignment


def conformer_lm_rescore(
    nbest: Nbest,
    model: torch.nn.Module,
    # TODO: add other arguments if needed
) -> k2.RaggedTensor:
    """Rescore an Nbest object with a conformer_lm model.

    Args:
      nbest:
        It contains linear FSAs to be rescored.
      model:
        A conformer lm model. See "conformer_lm/train.py"

    Returns:
      Return a ragged tensor containing scores for each path
      contained in the nbest. Its shape equals to `nbest.shape`.
    """
    assert hasattr(nbest.fsa, "tokens")
    # TODO:
