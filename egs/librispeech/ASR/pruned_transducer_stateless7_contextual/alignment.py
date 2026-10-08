# Copyright    2022-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Zengwei Yao)
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


from typing import List

import k2
import torch

from beam_search import Hypothesis, HypothesisList, get_hyps_shape

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
# Beam search is used to find the path with the highest log probabilities.
#
# It assumes the maximum number of symbols that can be
# emitted per frame is 1.


def batch_force_alignment(
    model: torch.nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ys_list: List[List[int]],
    beam_size: int = 4,
) -> List[int]:
    """Compute the force alignment of a batch of utterances given their transcripts
    in BPE tokens and the corresponding acoustic output from the encoder.

    Caution:
      This function is modified from `modified_beam_search` in beam_search.py.
      We assume that the maximum number of sybmols per frame is 1.

    Args:
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      ys_list:
        A list of BPE token IDs list. We require that for each utterance i,
        len(ys_list[i]) <= encoder_out_lens[i].
      beam_size:
        Size of the beam used in beam search.

    Returns:
      Return a list of frame indexes list for each utterance i,
      where len(ans[i]) == len(ys_list[i]).
    """
    assert encoder_out.ndim == 3, encoder_out.ndim
    assert encoder_out.size(0) == len(ys_list), (encoder_out.size(0), len(ys_list))
    assert encoder_out.size(0) > 0, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    sorted_indices = packed_encoder_out.sorted_indices.tolist()
    encoder_out_lens = encoder_out_lens.tolist()
    ys_lens = [len(ys) for ys in ys_list]
    sorted_encoder_out_lens = [encoder_out_lens[i] for i in sorted_indices]
    sorted_ys_lens = [ys_lens[i] for i in sorted_indices]
    sorted_ys_list = [ys_list[i] for i in sorted_indices]

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
        sorted_encoder_out_lens = sorted_encoder_out_lens[:batch_size]
        sorted_ys_lens = sorted_ys_lens[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out, decoder_out, project_input=False
        )  # (num_hyps, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)
        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(
            shape=log_probs_shape, value=log_probs.reshape(-1)
        )  # [batch][num_hyps*vocab_size]

        for i in range(batch_size):
            for h, hyp in enumerate(A[i]):
                pos_u = len(hyp.timestamp)
                idx_offset = h * vocab_size
                if (sorted_encoder_out_lens[i] - 1 - t) >= (sorted_ys_lens[i] - pos_u):
                    # emit blank token
                    new_hyp = Hypothesis(
                        log_prob=ragged_log_probs[i][idx_offset + blank_id],
                        ys=hyp.ys[:],
                        timestamp=hyp.timestamp[:],
                    )
                    B[i].add(new_hyp)
                if pos_u < sorted_ys_lens[i]:
                    # emit non-blank token
                    new_token = sorted_ys_list[i][pos_u]
                    new_hyp = Hypothesis(
                        log_prob=ragged_log_probs[i][idx_offset + new_token],
                        ys=hyp.ys + [new_token],
                        timestamp=hyp.timestamp + [t],
                    )
                    B[i].add(new_hyp)

            if len(B[i]) > beam_size:
                B[i] = B[i].topk(beam_size, length_norm=True)

    B = B + finalized_B
    sorted_hyps = [b.get_most_probable() for b in B]
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    hyps = [sorted_hyps[i] for i in unsorted_indices]
    ans = []
    for i, hyp in enumerate(hyps):
        assert hyp.ys[context_size:] == ys_list[i], (hyp.ys[context_size:], ys_list[i])
        ans.append(hyp.timestamp)

    return ans
