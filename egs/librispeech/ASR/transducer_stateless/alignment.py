# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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


from dataclasses import dataclass
from typing import List, Optional

import torch
from model import Transducer

# TODO(fangjun): Add more documentation

# The force alignment problem can be formulated as find
# a path in a rectangular lattice, where the path starts
# from the lower left corner and ends at the upper right
# corner. The horizontal axis of the lattice is `t`
# and the vertical axis is `u`.
#
# AlignItem is a node in the lattice, where its
# len(ys) equals to `t` and pos_u is the u coordinate
# in the lattice.
@dataclass
class AlignItem:
    log_prob: float
    ys: List[int]
    pos_u: int


class AlignItemList:
    def __init__(self, items: Optional[List[AlignItem]] = None):
        if items is None:
            items = []
        self.data = items

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> AlignItem:
        return self.data[i]

    def append(self, item: AlignItem) -> None:
        self.data.append(item)

    def get_active_items(self, T: int, U: int) -> "AlignItemList":
        ans = []
        for item in self:
            t = len(item.ys)
            if U - item.pos_u > T - t:
                continue
            ans.append(item)

        return AlignItemList(ans)

    def get_decoder_input(
        self,
        ys: List[int],
        context_size: int,
        blank_id: int,
    ) -> List[List[int]]:
        ans: List[List[int]] = []
        buf = [blank_id] * context_size + ys
        for item in self:
            # fmt: off
            ans.append(buf[item.pos_u:(item.pos_u + context_size)])
            # fmt: on
        return ans

    def topk(self, k: int) -> "AlignItemList":
        items = list(self)
        items = sorted(items, key=lambda i: i.log_prob, reverse=True)
        return AlignItemList(items[:k])


def force_alignment(
    model: Transducer,
    encoder_out: torch.Tensor,
    ys: List[int],
    beam_size: int = 4,
) -> List[int]:
    """
    Args:
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (N, T, C). Support only for N==1 now.
      ys:
        A list of token IDs. We require that len(ys) <= T.
      beam:
        Size of the beam used in beam search.
    Returns:
      Return a list of int such that
        - len(ans) == T
        - After removing blanks from ans, we have ans == ys.
    """
    import pdb

    pdb.set_trace()
    assert encoder_out.ndim == 3, encoder_out.ndim
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    assert 0 < len(ys) <= encoder_out.size(1), (len(ys), encoder_out.size(1))

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    T = encoder_out.size(1)
    U = len(ys)

    encoder_out_len = torch.tensor([1])
    decoder_out_len = encoder_out_len

    start = AlignItem(log_prob=0.0, ys=[], pos_u=0)
    B = AlignItemList([start])

    for t in range(T):
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # current_encoder_out is of shape (1, 1, encoder_out_dim)
        # fmt: on

        #  A = B.get_active_items()
        A = B  # shallow copy
        B = AlignItemList()

        decoder_input = A.get_decoder_input(
            ys=ys, context_size=context_size, blank_id=blank_id
        )
        decoder_input = torch.tensor(decoder_input, device=device)
        # decoder_input is of shape (num_active_items, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False)
        # decoder_output is of shape (num_active_items, 1, decoder_output_dim)

        current_encoder_out = current_encoder_out.expand(
            decoder_out.size(0), 1, -1
        )

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            encoder_out_len.expand(decoder_out.size(0)),
            decoder_out_len.expand(decoder_out.size(0)),
        )

        # logits is of shape (num_active_items, vocab_size)
        log_probs = logits.log_softmax(dim=-1).tolist()

        for i, item in enumerate(A):
            if (T - 1 - t) >= (U - item.pos_u):
                # horizontal transition
                new_item = AlignItem(
                    log_prob=item.log_prob + log_probs[i][blank_id],
                    ys=item.ys + [blank_id],
                    pos_u=item.pos_u,
                )
                B.append(new_item)

            if item.pos_u < U:
                # diagonal transition
                u = ys[item.pos_u]
                new_item = AlignItem(
                    log_prob=item.log_prob + log_probs[i][u],
                    ys=item.ys + [u],
                    pos_u=item.pos_u + 1,
                )
                B.append(new_item)
        if len(B) > beam_size:
            B = B.topk(beam_size)

    return B.topk(1)[0].ys
