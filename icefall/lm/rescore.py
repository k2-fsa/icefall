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

Suppose that an utterance has 3 paths:
    (a, b, c, d, e)

and we want to use a masked conformer LM to assign a likelihood to each path.

The following shows the steps:

(1) Select "a" as the reference path. Note: For ease of implementation,
    we always select the first path as the reference one.

(2) We have the following path pairs:

    (a, a) (a, b), (a, c), (a, d), (a, e)

    Note: Even if we know the likelihood for the pair (a, a) is always 0,
    we still list it here for ease of implementation.

(2) For each pair, e.g., for the pair (a, b),

    (i) Compute the alignment between "a" and "b"
    (ii) Use the computed alignment as `masked_src,`
    (iii) Use "a" as "src" and its shifted version as "tgt" (of course, we need
          to add bos and eos) and we can get a log-likelihood value (after
          negating the negative log-likelihood). Let us call this value as
          "ab_self"
    (iv)  Use "b" as "src" and its shifted version as "tgt".
          We can get another likelihood value, denoted as "ab_other"

So for the path pair (a, a), (a, b), (a, c), (a, d), and (a, e),
we can get the following log-likelihood values, viewed as two tensors:

    self = [aa_self, ab_self,   ac_self,  ad_self, ae_self]

    other = [aa_other, ab_other, ac_other, ad_other, ae_other]

    Compute the difference between the two tensors:

        other - self = [aa_other - aa_self, ab_other - ab_self,
                        ac_other - ac_self, ... ]

  The log-likelihood for path a is : 0
  The log-likelihood for path b is : ab_other - ab_self
  The log-likelihood for path c is : ac_other - ac_self
  The log-likelihood for path d is : ad_other - ad_self
  The log-likelihood for path e is : ae_other - ae_self

Note: "ab_other - ab_self" can be interpreted as

    log P(b) - log P(a)
"""

from typing import Tuple

import k2
import torch


def make_key_padding_mask(lengths: torch.Tensor):
    """
    TODO: add documentation

    >>> make_key_padding_mask(torch.tensor([3, 1, 4]))
    tensor([[False, False, False,  True],
            [False,  True,  True,  True],
            [False, False, False, False]])
    """
    assert lengths.dim() == 1

    bs = lengths.numel()
    max_len = lengths.max().item()
    device = lengths.device
    seq_range = torch.arange(0, max_len, device=device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)

    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


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
        size=(ragged.tot_size(0), 1),
        fill_value=value,
        device=device,
        dtype=dtype,
    )
    pad = k2.RaggedTensor(pad_values)

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


def make_hyp_to_ref_map(row_splits: torch.Tensor) -> torch.Tensor:
    """
    TODO: Add documentation

    >>> row_splits = torch.tensor([0, 3, 5], dtype=torch.int32)
    >>> make_hyp_to_ref_map(row_splits)
    tensor([0, 0, 0, 3, 3], dtype=torch.int32)
    """
    device = row_splits.device
    sizes = (row_splits[1:] - row_splits[:-1]).tolist()
    offsets = row_splits[:-1]

    map_tensor_list = []
    for size, offset in zip(sizes, offsets):
        # Explanation of the following operations
        # assume size is 3, offset is 2
        # torch.zeros() + offset is [2, 2, 2]
        map_tensor = (
            torch.zeros(size, dtype=torch.int32, device=device) + offset
        )
        map_tensor_list.append(map_tensor)

    return torch.cat(map_tensor_list)


def compute_alignment(
    tokens: k2.RaggedTensor,
    shape: k2.RaggedShape,
) -> k2.Fsa:
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

    hyps = k2.levenshtein_graph(tokens, device=device)

    refs = k2.levenshtein_graph(tokens, device=device)

    hyp_to_ref_map = make_hyp_to_ref_map(shape.row_splits(1))

    alignment = k2.levenshtein_alignment(
        refs=refs, hyps=hyps, hyp_to_ref_map=hyp_to_ref_map
    )
    return alignment


def prepare_conformer_lm_inputs(
    alignment: k2.Fsa,
    bos_id: int,
    eos_id: int,
    blank_id: int,
    src_label_name: str,
    unmasked_weight: float = 0.0,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    TODO: add documentation.

    Args:
      alignments:
        It is computed by :func:`compute_alignment`
      bos_id:
        ID of the bos symbol.
      eos_id:
        ID of the eos symbol.
      blank_id:
        ID of the blank symbol.
      src_label_name:
        The name of the attribute from `alignment` that will be used for `src`.
        `tgt` is a shift version of `src`. Valid values are: "ref_labels"
        and "hyp_labels".
    """
    assert src_label_name in ("ref_labels", "hyp_labels")
    device = alignment.device
    # alignment.arcs.shape has axes [fsa][state][arc]
    # we remove axis 1, i.e., state, here
    labels_shape = alignment.arcs.shape().remove_axis(1)

    masked_src = k2.RaggedTensor(labels_shape, alignment.labels.contiguous())
    masked_src = masked_src.remove_values_eq(-1)
    bos_masked_src = add_bos(masked_src, bos_id=bos_id)
    bos_masked_src_eos = add_eos(bos_masked_src, eos_id=eos_id)
    bos_masked_src_eos_pad = bos_masked_src_eos.pad(
        mode="constant", padding_value=blank_id
    )

    src = k2.RaggedTensor(labels_shape, getattr(alignment, src_label_name))
    src = src.remove_values_eq(-1)
    bos_src = add_bos(src, bos_id=bos_id)
    bos_src_eos = add_eos(bos_src, eos_id=eos_id)
    bos_src_eos_pad = bos_src_eos.pad(mode="constant", padding_value=blank_id)

    tgt = k2.RaggedTensor(labels_shape, getattr(alignment, src_label_name))
    # TODO: Do we need to remove 0s from tgt ?
    tgt = tgt.remove_values_eq(-1)
    tgt_eos = add_eos(tgt, eos_id=eos_id)

    # add a blank here since tgt_eos does not start with bos
    # assume blank id is 0
    tgt_eos = add_eos(tgt_eos, eos_id=blank_id)

    row_splits = tgt_eos.shape.row_splits(1)
    lengths = row_splits[1:] - row_splits[:-1]
    src_key_padding_mask = make_key_padding_mask(lengths)

    tgt_eos_pad = tgt_eos.pad(mode="constant", padding_value=blank_id)

    weight = torch.full(
        (tgt_eos_pad.size(0), tgt_eos_pad.size(1) - 1),
        fill_value=1,
        dtype=torch.float32,
        device=device,
    )

    # find unmasked positions
    unmasked_positions = bos_masked_src_eos_pad[:, 1:] != 0
    weight[unmasked_positions] = unmasked_weight

    # set weights for paddings
    weight[src_key_padding_mask[:, 1:]] = 0
    zeros = torch.zeros(weight.size(0), 1).to(weight)

    weight = torch.cat((weight, zeros), dim=1)

    # all other positions are assumed to be masked and
    # have the default weight 1

    return (
        bos_masked_src_eos_pad,
        bos_src_eos_pad,
        tgt_eos_pad,
        src_key_padding_mask,
        weight,
    )
