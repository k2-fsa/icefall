# Copyright      2025  Johns Hopkins University (author: Amir Hussein)
# Copyright    2022  Xiaomi Corp.        (authors: Wei Kang)
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

import warnings
from typing import List

import k2
import torch
import torch.nn as nn
from beam_search import Hypothesis, HypothesisList, get_hyps_shape
from decode_stream import DecodeStream

from icefall.decode import one_best_decoding
from icefall.utils import get_texts


def greedy_search_st(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_st: torch.Tensor,
    max_sym_per_frame: int,
    streams: List[DecodeStream],
    st_blank_penalty: float = 0.0,
) -> None:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      streams:
        A list of Stream objects.
    """
    assert len(streams) == encoder_out_st.size(0)
    assert encoder_out_st.ndim == 3
    # ST
    blank_id_st = model.st_decoder.blank_id
    context_size_st = model.st_decoder.context_size
    unk_id_st = getattr(model, "unk_id", blank_id_st)
    device = model.device
    T = encoder_out_st.size(1)
    # ASR
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id_st)

    #ST
    
    decoder_input_st = torch.tensor(
        [stream.hyp_st[-context_size_st:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    # decoder_out is of shape (N, 1, decoder_out_dim)
    decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False)
    decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
    
    # ASR
    decoder_input = torch.tensor(
        [stream.hyp_asr[-context_size:] for stream in streams],
        device=device,
        dtype=torch.int64,
    )
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # Maximum symbols per utterance.
    max_sym_per_utt = 10000
    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0
    t = 0
    # for t in range(T):
    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        # current_encoder_out_st = encoder_out_st[:, t : t + 1, :] # noqa
        current_encoder_out_st = encoder_out_st[:, t : t + 1, :].unsqueeze(2)

        st_logits = model.st_joiner(
            current_encoder_out_st,
            decoder_out_st.unsqueeze(1),
            project_input=False,
        )
        # logits'shape (batch_size,  vocab_size)
        st_logits = st_logits.squeeze(1).squeeze(1)
        if st_blank_penalty != 0.0:
            st_logits[:, 0] -= st_blank_penalty

        assert st_logits.ndim == 2, st_logits.shape
        y_st = st_logits.argmax(dim=1).tolist()
        for i, v in enumerate(y_st):
            if v not in (blank_id_st, unk_id_st):
                streams[i].hyp_st.append(v)
               
                # update decoder output
                # decoder_input_st = torch.tensor(
                #     [stream.hyp_st[-context_size_st:].reshape(
                # 1, context_size_st) for stream in streams],
                #     device=device,
                #     dtype=torch.int64,
                # )
                decoder_input_st = torch.stack([
                torch.tensor(stream.hyp_st[-context_size_st:], device=device, dtype=torch.int64)
                for stream in streams]).reshape(len(streams), context_size_st)
                
                decoder_out_st = model.st_decoder(
                    decoder_input_st,
                    need_pad=False,
                )
                decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
                sym_per_utt += 1
                sym_per_frame += 1
            else:
                sym_per_frame = 0
                t += 1


def modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    streams: List[DecodeStream],
    num_active_paths: int = 4,
    blank_penalty: float = 0.0,
) -> None:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The RNN-T model.
      encoder_out:
        A 3-D tensor of shape (N, T, encoder_out_dim) containing the output of
        the encoder model.
      streams:
        A list of stream objects.
      num_active_paths:
        Number of active paths during the beam search.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert len(streams) == encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    batch_size = len(streams)
    T = encoder_out.size(1)

    B = [stream.hyps for stream in streams]

    for t in range(T):
        current_encoder_out = encoder_out[:, t].unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.stack(
            [hyp.log_prob.reshape(1) for hyps in A for hyp in hyps], dim=0
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_out is of shape (num_hyps, 1, 1, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, encoder_out_dim)

        logits = model.joiner(current_encoder_out, decoder_out, project_input=False)
        # logits is of shape (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0.0:
            logits[:, 0] -= blank_penalty

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(num_active_paths)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                if new_token != blank_id:
                    new_ys.append(new_token)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                B[i].add(new_hyp)

    for i in range(batch_size):
        streams[i].hyps = B[i]