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
from typing import Iterator, List, Optional

import sentencepiece as spm
import torch
from model import Transducer

# The force alignment problem can be formulated as finding
# a path in a rectangular lattice, where the path starts
# from the lower left corner and ends at the upper right
# corner. The horizontal axis of the lattice is `t` (representing
# acoustic frame indexes) and the vertical axis is `u` (representing
# BPE tokens of the transcript).
#
# The notations `t` and `u` are from the paper
# https://arxiv.org/pdf/1211.3711.pdf
#
# Beam search is used to find the path with the
# highest log probabilities.
#
# It assumes the maximum number of symbols that can be
# emitted per frame is 1. You can use `--modified-transducer-prob`
# from `./train.py` to train a model that satisfies this assumption.


# AlignItem is the ending node of a path originated from the starting node.
# len(ys) equals to `t` and pos_u is the u coordinate
# in the lattice.
@dataclass
class AlignItem:
    # total log prob of the path that ends at this item.
    # The path is originated from the starting node.
    log_prob: float

    # It contains framewise token alignment
    ys: List[int]

    # It equals to the number of non-zero entries in ys
    pos_u: int


class AlignItemList:
    def __init__(self, items: Optional[List[AlignItem]] = None):
        """
        Args:
          items:
            A list of AlignItem
        """
        if items is None:
            items = []
        self.data = items

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of AlignItem in this object."""
        return len(self.data)

    def __getitem__(self, i: int) -> AlignItem:
        """Return the i-th item in this object."""
        return self.data[i]

    def append(self, item: AlignItem) -> None:
        """Append an item to the end of this object."""
        self.data.append(item)

    def get_decoder_input(
        self,
        ys: List[int],
        context_size: int,
        blank_id: int,
    ) -> List[List[int]]:
        """Get input for the decoder for each item in this object.

        Args:
          ys:
            The transcript of the utterance in BPE tokens.
          context_size:
            Context size of the NN decoder model.
          blank_id:
            The ID of the blank symbol.
        Returns:
          Return a list-of-list int. `ans[i]` contains the decoder
          input for the i-th item in this object and its lengths
          is `context_size`.
        """
        ans: List[List[int]] = []
        buf = [blank_id] * context_size + ys
        for item in self:
            # fmt: off
            ans.append(buf[item.pos_u:(item.pos_u + context_size)])
            # fmt: on
        return ans

    def topk(self, k: int) -> "AlignItemList":
        """Return the top-k items.

        Items are ordered by their log probs in descending order
        and the top-k items are returned.

        Args:
          k:
            Size of top-k.
        Returns:
          Return a new AlignItemList that contains the top-k items
          in this object. Caution: It uses shallow copy.
        """
        items = list(self)
        items = sorted(items, key=lambda i: i.log_prob, reverse=True)
        return AlignItemList(items[:k])


def force_alignment(
    model: Transducer,
    encoder_out: torch.Tensor,
    ys: List[int],
    beam_size: int = 4,
) -> List[int]:
    """Compute the force alignment of an utterance given its transcript
    in BPE tokens and the corresponding acoustic output from the encoder.

    Caution:
      We assume that the maximum number of sybmols per frame is 1.
      That is, the model should be trained using a nonzero value
      for the option `--modified-transducer-prob` in train.py.

    Args:
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (N, T, C). Support only for N==1 at present.
      ys:
        A list of BPE token IDs. We require that len(ys) <= T.
      beam_size:
        Size of the beam used in beam search.
    Returns:
      Return a list of int such that
        - len(ans) == T
        - After removing blanks from ans, we have ans == ys.
    """
    assert encoder_out.ndim == 3, encoder_out.ndim
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    assert 0 < len(ys) <= encoder_out.size(1), (len(ys), encoder_out.size(1))

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    T = encoder_out.size(1)
    U = len(ys)
    assert 0 < U <= T

    encoder_out_len = torch.tensor([1])
    decoder_out_len = encoder_out_len

    start = AlignItem(log_prob=0.0, ys=[], pos_u=0)
    B = AlignItemList([start])

    for t in range(T):
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # current_encoder_out is of shape (1, 1, encoder_out_dim)
        # fmt: on

        A = B  # shallow copy
        B = AlignItemList()

        decoder_input = A.get_decoder_input(
            ys=ys, context_size=context_size, blank_id=blank_id
        )
        decoder_input = torch.tensor(decoder_input, device=device)
        # decoder_input is of shape (num_active_items, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False)
        # decoder_output is of shape (num_active_items, 1, decoder_output_dim)

        current_encoder_out = current_encoder_out.expand(decoder_out.size(0), 1, -1)

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
                # horizontal transition (left -> right)
                new_item = AlignItem(
                    log_prob=item.log_prob + log_probs[i][blank_id],
                    ys=item.ys + [blank_id],
                    pos_u=item.pos_u,
                )
                B.append(new_item)

            if item.pos_u < U:
                # diagonal transition (lower left -> upper right)
                u = ys[item.pos_u]
                new_item = AlignItem(
                    log_prob=item.log_prob + log_probs[i][u],
                    ys=item.ys + [u],
                    pos_u=item.pos_u + 1,
                )
                B.append(new_item)

        if len(B) > beam_size:
            B = B.topk(beam_size)

    ans = B.topk(1)[0].ys

    assert len(ans) == T
    assert list(filter(lambda i: i != blank_id, ans)) == ys

    return ans


def get_word_starting_frames(
    ali: List[int], sp: spm.SentencePieceProcessor
) -> List[int]:
    """Get the starting frame of each word from the given token alignments.

    When a word is encoded into BPE tokens, the first token starts
    with underscore "_", which can be used to identify the starting frame
    of a word.

    Args:
      ali:
        Framewise token alignment. It can be the return value of
        :func:`force_alignment`.
      sp:
        The sentencepiece model.
    Returns:
      Return a list of int representing the starting frame of each word
      in the alignment.
      Caution:
        You have to take into account the model subsampling factor when
        converting the starting frame into time.
    """
    underscore = b"\xe2\x96\x81".decode()  # '_'
    ans = []
    for i in range(len(ali)):
        if sp.id_to_piece(ali[i]).startswith(underscore):
            ans.append(i)
    return ans
