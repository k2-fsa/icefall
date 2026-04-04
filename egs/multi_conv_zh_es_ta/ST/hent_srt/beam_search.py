# Copyright      2025  Johns Hopkins University (author: Amir Hussein)
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang
#                                                  Xiaoyu Yang)
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
from torch import nn

from icefall import ContextGraph, ContextState, NgramLm, NgramLmStateCost
from icefall.decode import Nbest, one_best_decoding
from icefall.lm_wrapper import LmScorer
from icefall.rnn_lm.model import RnnLmModel
from icefall.transformer_lm.model import TransformerLM
from icefall.utils import (
    DecodingResults,
    add_eos,
    add_sos,
    get_texts,
    get_texts_with_timestamp,
)


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]

    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        # If logit for blank token is positive, the output should be blank (Bernoulli)
        y = torch.zeros_like(logits[:, 0], dtype=torch.int64, device=device)
        # If logit for blank token is negative, the output should be the argmax
        # of the rest of the logits
        y += torch.where(logits[:, 0] <= 0, logits[:, 1:].argmax(dim=1) + 1, 0)
        # Convert y to list
        y = y.tolist()

        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
            scores=ans_scores,
        )

def greedy_search_st(
    model: nn.Module,
    encoder_out: torch.Tensor,
    st_encoder_out: torch.Tensor,
    max_sym_per_frame: int,
    max_sym_per_frame_asr:int=1,
    st_blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert st_encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert st_encoder_out.size(0) == 1, st_encoder_out.size(0)

    blank_id_st = model.st_decoder.blank_id
    context_size_st = model.st_decoder.context_size
    unk_id_st = getattr(model, "unk_id", blank_id_st)
    
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id_st)

    device = next(model.parameters()).device

    decoder_input_st = torch.tensor(
        [-1] * (context_size_st - 1) + [blank_id_st], device=device, dtype=torch.int64
    ).reshape(1, context_size_st)
    
    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], device=device, dtype=torch.int64
    ).reshape(1, context_size)
    
    decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False)
    decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
    encoder_out_st = model.st_joiner.encoder_proj(st_encoder_out)    

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out_st.size(1)
    t = 0
    hyp_st = [blank_id_st] * context_size_st

    # timestamp[i] is the frame index after subsampling
    # on which hyp[i] is decoded
    timestamp_st = []

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000

    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        # fmt: off
        current_encoder_out_st = encoder_out_st[:, t:t+1, :].unsqueeze(2)
        # fmt: on
        logits_st = model.st_joiner(
            current_encoder_out_st, decoder_out_st.unsqueeze(1), project_input=False
        )
        # logits is (1, 1, 1, vocab_size)

        if st_blank_penalty != 0:
            logits_st[:, :, :, 0] -= st_blank_penalty

        y_st = logits_st.argmax().item()
        if y_st not in (blank_id_st, unk_id_st):
            hyp_st.append(y_st)
            timestamp_st.append(t)
            decoder_input_st = torch.tensor([hyp_st[-context_size_st:]], device=device).reshape(
                1, context_size_st
            )

            decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False)
            decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
        
    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # timestamp[i] is the frame index after subsampling
    # on which hyp[i] is decoded
    timestamp = []

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000

    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame_asr:
            sym_per_frame = 0
            t += 1
            continue

        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        # fmt: on
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits is (1, 1, 1, vocab_size)

        y = logits.argmax().item()
        if y not in (blank_id, unk_id):
            hyp.append(y)
            timestamp.append(t)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )

            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
            
    hyp_st = hyp_st[context_size_st:]
    hyp = hyp[context_size:]  # remove blanks

    if not return_timestamps:
        return [hyp, hyp_st]
    else:
        return DecodingResults(
            hyps=[hyp, hyp_st],
            timestamps=[timestamp, timestamp_st],
        )

def greedy_search_batch_st(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    st_encoder_out: torch.Tensor = None,
    st_encoder_out_lens: torch.Tensor = None,
    st_blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert st_encoder_out.ndim == 3, st_encoder_out.shape
    assert st_encoder_out.size(0) >= 1, st_encoder_out.size(0)
    # ST
    packed_st_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=st_encoder_out,
        lengths=st_encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    device = next(model.parameters()).device
    blank_id_st = model.st_decoder.blank_id
    unk_id_st = getattr(model, "unk_id", blank_id_st)
    context_size_st = model.st_decoder.context_size
    
    
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    

    batch_size_list = packed_st_encoder_out.batch_sizes.tolist()
    N = st_encoder_out.size(0)
    assert torch.all(st_encoder_out_lens > 0), st_encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)
    
    hyps_st = [[-1] * (context_size_st - 1) + [blank_id_st] for _ in range(N)]

    timestamps_st = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores_st = [[] for _ in range(N)]

    decoder_input_st = torch.tensor(
        hyps_st,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False)
    decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out_st = model.st_joiner.encoder_proj(packed_st_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out_st = encoder_out_st.data[start:end]
        current_encoder_out_st = current_encoder_out_st.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out_st = decoder_out_st[:batch_size]
        logits_st = model.st_joiner(
            current_encoder_out_st, decoder_out_st.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits_st = logits_st.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits_st.ndim == 2, logits_st.shape

        if st_blank_penalty != 0:
            logits_st[:, 0] -= st_blank_penalty

        # If logit for blank token is positive, the output should be blank (Bernoulli)
        y_st = logits_st.argmax(dim=1).tolist()

        emitted = False
        for i, v in enumerate(y_st):
            if v not in (blank_id_st, unk_id_st):
                hyps_st[i].append(v)
                timestamps_st[i].append(t)
                scores_st[i].append(logits_st[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input_st = [h[-context_size_st:] for h in hyps_st[:batch_size]]
            decoder_input_st = torch.tensor(
                decoder_input_st,
                device=device,
                dtype=torch.int64,
            )
            decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False)
            decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
    
    # ASR
    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]
    # timestamp[n][i] is the frame index after subsampling
    # on which hyp[n][i] is decoded
    timestamps = [[] for _ in range(N)]
    # scores[n][i] is the logits on which hyp[n][i] is decoded
    scores = [[] for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]
        logits = model.joiner(
            current_encoder_out, decoder_out.unsqueeze(1), project_input=False
        )
        # logits'shape (batch_size, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (batch_size, vocab_size)
        assert logits.ndim == 2, logits.shape

        # If logit for blank token is positive, the output should be blank (Bernoulli)
        y = logits.argmax(dim=1).tolist()

        emitted = False
        for i, v in enumerate(y):
            if v not in (blank_id, unk_id):
                hyps[i].append(v)
                timestamps[i].append(t)
                scores[i].append(logits[i, v].item())
                emitted = True
        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )
            decoder_out = model.decoder(decoder_input, need_pad=False)
            decoder_out = model.joiner.decoder_proj(decoder_out)
    # ST
    sorted_ans_st = [h[context_size_st:] for h in hyps_st]
    ans_st = []
    ans_timestamps_st = []
    ans_scores_st = []
    unsorted_indices_st = packed_st_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans_st.append(sorted_ans_st[unsorted_indices_st[i]])
        ans_timestamps_st.append(timestamps_st[unsorted_indices_st[i]])
        ans_scores_st.append(scores_st[unsorted_indices_st[i]])
    
    # ASR
    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    ans_timestamps = []
    ans_scores = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(timestamps[unsorted_indices[i]])
        ans_scores.append(scores[unsorted_indices[i]])


    if not return_timestamps:
        return [ans, ans_st]
    else:
        return DecodingResults(
            hyps=[ans, ans_st],
            timestamps=[ans_timestamps, ans_timestamps_st]
            # scores=[ans_scores, ans_scores_st],
        )
        
@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

    # timestamp[i] is the frame index after subsampling
    # on which ys[i] is decoded
    timestamp: List[int] = field(default_factory=list)

    # the lm score for next token given the current ys
    lm_score: Optional[torch.Tensor] = None

    # the RNNLM states (h and c in LSTM)
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # N-gram LM state
    state_cost: Optional[NgramLmStateCost] = None

    # Context graph state
    context_state: Optional[ContextState] = None
    
    decoder_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    

    @property
    def key(self) -> str:
        """Return a string representation of self.ys"""
        return "_".join(map(str, self.ys))


class HypothesisList(object):
    def __init__(self, data: Optional[Dict[str, Hypothesis]] = None) -> None:
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
    def data(self) -> Dict[str, Hypothesis]:
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
            torch.logaddexp(old_hyp.log_prob, hyp.log_prob, out=old_hyp.log_prob)
        else:
            self._data[key] = hyp

    def get_most_probable(self, length_norm: bool = False) -> Hypothesis:
        """Get the most probable hypothesis, i.e., the one with
        the largest `log_prob`.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        Returns:
          Return the hypothesis that has the largest `log_prob`.
        """
        if length_norm:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob / len(hyp.ys))
        else:
            return max(self._data.values(), key=lambda hyp: hyp.log_prob)

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
        """Remove all Hypotheses whose log_prob is less than threshold.

        Caution:
          `self` is not modified. Instead, a new HypothesisList is returned.

        Returns:
          Return a new HypothesisList containing all hypotheses from `self`
          with `log_prob` being greater than the given `threshold`.
        """
        ans = HypothesisList()
        for _, hyp in self._data.items():
            if hyp.log_prob > threshold:
                ans.add(hyp)  # shallow copy
        return ans

    def topk(self, k: int, length_norm: bool = False) -> "HypothesisList":
        """Return the top-k hypothesis.

        Args:
          length_norm:
            If True, the `log_prob` of a hypothesis is normalized by the
            number of tokens in it.
        """
        hyps = list(self._data.items())

        if length_norm:
            hyps = sorted(
                hyps, key=lambda h: h[1].log_prob / len(h[1].ys), reverse=True
            )[:k]
        else:
            hyps = sorted(hyps, key=lambda h: h[1].log_prob, reverse=True)[:k]

        ans = HypothesisList(dict(hyps))
        return ans

    def __contains__(self, key: str):
        return key in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        s = []
        for key in self:
            s.append(key)
        return ", ".join(s)


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


def modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    st_encoder_out: torch.Tensor = None,
    st_encoder_out_lens: torch.Tensor = None,
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
    use_hat: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    packed_st_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=st_encoder_out,
        lengths=st_encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    blank_id_st = model.st_decoder.blank_id
    unk_id_st = getattr(model, "unk_id", blank_id_st)

    context_size = model.decoder.context_size
    context_size_st = model.st_decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )
    B_st = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B_st[i].add(
            Hypothesis(
                ys=[blank_id_st] * context_size_st,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )
    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)
    encoder_out_st = model.st_joiner.encoder_proj(packed_st_encoder_out.data)

    offset = 0
    finalized_B = []
    finalized_B_st = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        current_encoder_out_st = encoder_out_st.data[start:end]
        current_encoder_out_st = current_encoder_out_st.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
        hyps_shape = get_hyps_shape(B).to(device)
        
        finalized_B_st = B_st[batch_size:] + finalized_B_st
        B_st = B_st[:batch_size]

        hyps_shape_st = get_hyps_shape(B_st).to(device)

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

        A_st = [list(b) for b in B_st]
        B_st = [HypothesisList() for _ in range(batch_size)]
        ys_log_probs_st = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A_st for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input_st = torch.tensor(
            [hyp.ys[-context_size_st:] for hyps in A_st for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False).unsqueeze(1)
        decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        # For blank symbol, log-prob is log-sigmoid of the score
        if use_hat == True:
          # For blank symbol, log-prob is log-sigmoid of the score
          logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
          # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
          # need to add the following term to the log-probs of non-blank symbols. This
          # is equivalent to log(1 - sigmoid(logits[..., 0])).
          #breakpoint()
          nb_shift = logp_b - logits[..., 0]
          nb_shift = nb_shift.unsqueeze(-1)
          log_probs1 = (logits[..., 1:] / temperature).log_softmax(dim=-1) + nb_shift # (num_hyps, vocab_size-1)
          log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1) 
          log_probs.add_(ys_log_probs)
        else:
          log_probs = (logits / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
          log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)


        # st
        current_encoder_out_st = torch.index_select(
            current_encoder_out_st,
            dim=0,
            index=hyps_shape_st.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits_st = model.st_joiner(
            current_encoder_out_st,
            decoder_out_st,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits_st = logits_st.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        if use_hat == True:
            # For blank symbol, log-prob is log-sigmoid of the score
            logp_b_st = torch.nn.functional.logsigmoid(logits_st[..., 0])
            # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
            # need to add the following term to the log-probs of non-blank symbols. This
            # is equivalent to log(1 - sigmoid(logits[..., 0])).
            nb_shift_st = logp_b_st - logits_st[..., 0]
            nb_shift_st = nb_shift_st.unsqueeze(-1)
            log_probs1_st = (logits_st[..., 1:] / temperature).log_softmax(
                dim=-1
            ) + nb_shift_st  # (num_hyps, vocab_size-1)
            log_probs_st = torch.cat((logp_b_st.unsqueeze(-1), log_probs1_st), dim=-1)
            log_probs_st.add_(ys_log_probs_st)
        else:
            log_probs_st = (logits_st / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
            log_probs_st.add_(ys_log_probs_st)
            
        vocab_size_st = log_probs_st.size(-1)

        log_probs_st = log_probs_st.reshape(-1)
        # print(hyps_shape_st.row_splits(1) * vocab_size_st)
        # print(log_probs_st.numel())
        row_splits_st = hyps_shape_st.row_splits(1) * vocab_size_st
        log_probs_shape_st = k2.ragged.create_ragged_shape2(
            row_splits=row_splits_st, cached_tot_size=log_probs_st.numel()
        )
        ragged_log_probs_st = k2.RaggedTensor(shape=log_probs_shape_st, value=log_probs_st)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)
            topk_log_probs_st, topk_indexes_st = ragged_log_probs_st[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

                topk_hyp_indexes_st = (topk_indexes_st // vocab_size_st).tolist()
                topk_token_indexes_st = (topk_indexes_st % vocab_size_st).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

            for k in range(len(topk_hyp_indexes_st)):
                hyp_idx = topk_hyp_indexes_st[k]
                hyp_st = A_st[i][hyp_idx]

                new_ys_st = hyp_st.ys[:]
                new_token = topk_token_indexes_st[k]
                new_timestamp_st = hyp_st.timestamp[:]
                if new_token not in (blank_id_st, unk_id_st):
                    new_ys_st.append(new_token)
                    new_timestamp_st.append(t)

                new_log_prob_st = topk_log_probs_st[k]
                new_hyp_st = Hypothesis(
                    ys=new_ys_st, log_prob=new_log_prob_st, timestamp=new_timestamp_st
                )
                B_st[i].add(new_hyp_st)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    B_st = B_st + finalized_B_st
    best_hyps_st = [b.get_most_probable(length_norm=True) for b in B_st]

    sorted_ans_st = [h.ys[context_size:] for h in best_hyps_st]
    sorted_timestamps_st = [h.timestamp for h in best_hyps_st]
    ans_st = []
    ans_timestamps_st = []
    st_unsorted_indices = packed_st_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans_st.append(sorted_ans_st[st_unsorted_indices[i]])
        ans_timestamps_st.append(sorted_timestamps_st[st_unsorted_indices[i]])

    if not return_timestamps:
        return [ans, ans_st]
    else:
        return DecodingResults(
            hyps=[ans, ans_st],
            timestamps=[ans_timestamps, ans_timestamps_st],
        )


def modified_beam_search_st(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    st_encoder_out: torch.Tensor = None,
    st_encoder_out_lens: torch.Tensor = None,
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
    use_hat: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert st_encoder_out.ndim == 3, st_encoder_out.shape
    assert st_encoder_out.size(0) >= 1, st_encoder_out.size(0)

    packed_st_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=st_encoder_out,
        lengths=st_encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id_st = model.st_decoder.blank_id
    unk_id_st = getattr(model, "unk_id", blank_id_st)

    context_size_st = model.st_decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_st_encoder_out.batch_sizes.tolist()
    N = st_encoder_out.size(0)
    assert torch.all(st_encoder_out_lens > 0), st_encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B_st = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B_st[i].add(
            Hypothesis(
                ys=[blank_id_st] * context_size_st,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )
    encoder_out_st = model.st_joiner.encoder_proj(packed_st_encoder_out.data)

    offset = 0
    finalized_B_st = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out_st = encoder_out_st.data[start:end]
        current_encoder_out_st = current_encoder_out_st.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end
        
        finalized_B_st = B_st[batch_size:] + finalized_B_st
        B_st = B_st[:batch_size]

        hyps_shape_st = get_hyps_shape(B_st).to(device)

        A_st = [list(b) for b in B_st]
        B_st = [HypothesisList() for _ in range(batch_size)]
        ys_log_probs_st = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A_st for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input_st = torch.tensor(
            [hyp.ys[-context_size_st:] for hyps in A_st for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False).unsqueeze(1)
        decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)
        # st
        current_encoder_out_st = torch.index_select(
            current_encoder_out_st,
            dim=0,
            index=hyps_shape_st.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits_st = model.st_joiner(
            current_encoder_out_st,
            decoder_out_st,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits_st = logits_st.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        if use_hat == True:
            # For blank symbol, log-prob is log-sigmoid of the score
            logp_b_st = torch.nn.functional.logsigmoid(logits_st[..., 0])
            # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
            # need to add the following term to the log-probs of non-blank symbols. This
            # is equivalent to log(1 - sigmoid(logits[..., 0])).
            nb_shift_st = logp_b_st - logits_st[..., 0]
            nb_shift_st = nb_shift_st.unsqueeze(-1)
            log_probs1_st = (logits_st[..., 1:] / temperature).log_softmax(
                dim=-1
            ) + nb_shift_st  # (num_hyps, vocab_size-1)
            log_probs_st = torch.cat((logp_b_st.unsqueeze(-1), log_probs1_st), dim=-1)
            log_probs_st.add_(ys_log_probs_st)
        else:
            log_probs_st = (logits_st / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
            log_probs_st.add_(ys_log_probs_st)
            
        vocab_size_st = log_probs_st.size(-1)

        log_probs_st = log_probs_st.reshape(-1)
        # print(hyps_shape_st.row_splits(1) * vocab_size_st)
        # print(log_probs_st.numel())
        row_splits_st = hyps_shape_st.row_splits(1) * vocab_size_st
        log_probs_shape_st = k2.ragged.create_ragged_shape2(
            row_splits=row_splits_st, cached_tot_size=log_probs_st.numel()
        )
        ragged_log_probs_st = k2.RaggedTensor(shape=log_probs_shape_st, value=log_probs_st)

        for i in range(batch_size):
            topk_log_probs_st, topk_indexes_st = ragged_log_probs_st[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                topk_hyp_indexes_st = (topk_indexes_st // vocab_size_st).tolist()
                topk_token_indexes_st = (topk_indexes_st % vocab_size_st).tolist()

            for k in range(len(topk_hyp_indexes_st)):
                hyp_idx = topk_hyp_indexes_st[k]
                hyp_st = A_st[i][hyp_idx]

                new_ys_st = hyp_st.ys[:]
                new_token = topk_token_indexes_st[k]
                new_timestamp_st = hyp_st.timestamp[:]
                if new_token not in (blank_id_st, unk_id_st):
                    new_ys_st.append(new_token)
                    new_timestamp_st.append(t)

                new_log_prob_st = topk_log_probs_st[k]
                new_hyp_st = Hypothesis(
                    ys=new_ys_st, log_prob=new_log_prob_st, timestamp=new_timestamp_st
                )
                B_st[i].add(new_hyp_st)


    B_st = B_st + finalized_B_st
    best_hyps_st = [b.get_most_probable(length_norm=True) for b in B_st]

    sorted_ans_st = [h.ys[context_size_st:] for h in best_hyps_st]
    sorted_timestamps_st = [h.timestamp for h in best_hyps_st]
    ans_st = []
    ans_timestamps_st = []
    st_unsorted_indices = packed_st_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans_st.append(sorted_ans_st[st_unsorted_indices[i]])
        ans_timestamps_st.append(sorted_timestamps_st[st_unsorted_indices[i]])

    if not return_timestamps:
        return [None, ans_st]
    else:
        return DecodingResults(
            hyps=[None, ans_st],
            timestamps=[None, ans_timestamps_st],
        )
        

def modified_beam_search_st2(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    st_encoder_out: torch.Tensor = None,
    st_encoder_out_lens: torch.Tensor = None,
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
    use_hat: bool = False,
    st_blank_penalty: float = 0,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert st_encoder_out.ndim == 3, st_encoder_out.shape
    assert st_encoder_out.size(0) >= 1, st_encoder_out.size(0)

    packed_st_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=st_encoder_out,
        lengths=st_encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id_st = model.st_decoder.blank_id
    unk_id_st = getattr(model, "unk_id", blank_id_st)

    context_size_st = model.st_decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_st_encoder_out.batch_sizes.tolist()
    N = st_encoder_out.size(0)
    assert torch.all(st_encoder_out_lens > 0), st_encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B_st = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B_st[i].add(
            Hypothesis(
                ys=[blank_id_st] * context_size_st,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )
    encoder_out_st = model.st_joiner.encoder_proj(packed_st_encoder_out.data)

    offset = 0
    finalized_B_st = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out_st = encoder_out_st.data[start:end]
        current_encoder_out_st = current_encoder_out_st.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end
        
        finalized_B_st = B_st[batch_size:] + finalized_B_st
        B_st = B_st[:batch_size]

        hyps_shape_st = get_hyps_shape(B_st).to(device)

        A_st = [list(b) for b in B_st]
        B_st = [HypothesisList() for _ in range(batch_size)]
        ys_log_probs_st = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A_st for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input_st = torch.tensor(
            [hyp.ys[-context_size_st:] for hyps in A_st for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)
       
        decoder_out_st = model.st_decoder(decoder_input_st, need_pad=False).unsqueeze(1)
        decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)
        # st
        current_encoder_out_st = torch.index_select(
            current_encoder_out_st,
            dim=0,
            index=hyps_shape_st.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits_st = model.st_joiner(
            current_encoder_out_st,
            decoder_out_st,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits_st = logits_st.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        if st_blank_penalty != 0:
            logits_st[:, 0] -= st_blank_penalty
        if use_hat == True:
            # For blank symbol, log-prob is log-sigmoid of the score
            logp_b_st = torch.nn.functional.logsigmoid(logits_st[..., 0])
            # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
            # need to add the following term to the log-probs of non-blank symbols. This
            # is equivalent to log(1 - sigmoid(logits[..., 0])).
            nb_shift_st = logp_b_st - logits_st[..., 0]
            nb_shift_st = nb_shift_st.unsqueeze(-1)
            log_probs1_st = (logits_st[..., 1:] / temperature).log_softmax(
                dim=-1
            ) + nb_shift_st  # (num_hyps, vocab_size-1)
            log_probs_st = torch.cat((logp_b_st.unsqueeze(-1), log_probs1_st), dim=-1)
            log_probs_st.add_(ys_log_probs_st)
        else:
            log_probs_st = (logits_st / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
            log_probs_st.add_(ys_log_probs_st)
            
        vocab_size_st = log_probs_st.size(-1)

        log_probs_st = log_probs_st.reshape(-1)
        # print(hyps_shape_st.row_splits(1) * vocab_size_st)
        # print(log_probs_st.numel())
        row_splits_st = hyps_shape_st.row_splits(1) * vocab_size_st
        log_probs_shape_st = k2.ragged.create_ragged_shape2(
            row_splits=row_splits_st, cached_tot_size=log_probs_st.numel()
        )
        ragged_log_probs_st = k2.RaggedTensor(shape=log_probs_shape_st, value=log_probs_st)

        for i in range(batch_size):
            topk_log_probs_st, topk_indexes_st = ragged_log_probs_st[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                topk_hyp_indexes_st = (topk_indexes_st // vocab_size_st).tolist()
                topk_token_indexes_st = (topk_indexes_st % vocab_size_st).tolist()

            for k in range(len(topk_hyp_indexes_st)):
                hyp_idx = topk_hyp_indexes_st[k]
                hyp_st = A_st[i][hyp_idx]

                new_ys_st = hyp_st.ys[:]
                new_token = topk_token_indexes_st[k]
                new_timestamp_st = hyp_st.timestamp[:]
                if new_token not in (blank_id_st, unk_id_st):
                    new_ys_st.append(new_token)
                    new_timestamp_st.append(t)

                new_log_prob_st = topk_log_probs_st[k]
                new_hyp_st = Hypothesis(
                    ys=new_ys_st, log_prob=new_log_prob_st, timestamp=new_timestamp_st
                )
                B_st[i].add(new_hyp_st)

    # ASR
    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    batch_size_list = packed_encoder_out.batch_sizes.tolist()
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
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
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
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        # For blank symbol, log-prob is log-sigmoid of the score
        if use_hat == True:
          # For blank symbol, log-prob is log-sigmoid of the score
          logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
          # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
          # need to add the following term to the log-probs of non-blank symbols. This
          # is equivalent to log(1 - sigmoid(logits[..., 0])).
          #breakpoint()
          nb_shift = logp_b - logits[..., 0]
          nb_shift = nb_shift.unsqueeze(-1)
          log_probs1 = (logits[..., 1:] / temperature).log_softmax(dim=-1) + nb_shift # (num_hyps, vocab_size-1)
          log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1) 
          log_probs.add_(ys_log_probs)
        else:
          log_probs = (logits / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
          log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
    
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    
    B_st = B_st + finalized_B_st
    best_hyps_st = [b.get_most_probable(length_norm=True) for b in B_st]

    sorted_ans_st = [h.ys[context_size_st:] for h in best_hyps_st]
    sorted_timestamps_st = [h.timestamp for h in best_hyps_st]
    ans_st = []
    ans_timestamps_st = []
    st_unsorted_indices = packed_st_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans_st.append(sorted_ans_st[st_unsorted_indices[i]])
        ans_timestamps_st.append(sorted_timestamps_st[st_unsorted_indices[i]])

    if not return_timestamps:
        return [ans, ans_st]
    else:
        return DecodingResults(
            hyps=[ans, ans_st],
            timestamps=[ans_timestamps, ans_timestamps_st],
        )


def modified_beam_search_st2_lstm(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    st_encoder_out: torch.Tensor = None,
    st_encoder_out_lens: torch.Tensor = None,
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
    use_hat: bool = False,
    st_lstm_pred: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert st_encoder_out.ndim == 3, st_encoder_out.shape
    assert st_encoder_out.size(0) >= 1, st_encoder_out.size(0)

    packed_st_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=st_encoder_out,
        lengths=st_encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id_st = model.st_decoder.blank_id
    unk_id_st = getattr(model, "unk_id", blank_id_st)
    device = next(model.parameters()).device

    batch_size_list = packed_st_encoder_out.batch_sizes.tolist()
    N = st_encoder_out.size(0)
    assert torch.all(st_encoder_out_lens > 0), st_encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)
    init_in = torch.tensor([blank_id_st], device=device).reshape(1, 1)
    decoder_out, (h, c) = model.st_decoder(init_in)
    B_st = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B_st[i].add(
            Hypothesis(
                ys=[blank_id_st],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
                # decoder_state=None,
                decoder_state=[(h, c)],
            )
        )
    encoder_out_st = model.st_joiner.encoder_proj(packed_st_encoder_out.data)

    offset = 0
    finalized_B_st = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out_st = encoder_out_st.data[start:end]
        current_encoder_out_st = current_encoder_out_st.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end
        # breakpoint()
        finalized_B_st = B_st[batch_size:] + finalized_B_st
        B_st = B_st[:batch_size]   
        hyps_shape_st = get_hyps_shape(B_st).to(device)

        A_st = [list(b) for b in B_st]
        B_st = [HypothesisList() for _ in range(batch_size)]
        ys_log_probs_st = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A_st for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input_st = torch.tensor(
                    [hyp.ys[-1] for hyps in A_st for hyp in hyps],
                        device=device,
                        dtype=torch.int64,
                        ).unsqueeze(1)  # (num_hyps, 1)
        
        st_decode_state = [hyp.decoder_state[-1] for hyps in A_st for hyp in hyps]
        cur_h_states_st = torch.cat([state[0] for state in st_decode_state], dim=1)  # (num_layers, batch, hidden_dim)
        cur_c_states_st = torch.cat([state[1] for state in st_decode_state], dim=1)  # (num_layers, batch, hidden_dim)
        st_decode_state = (cur_h_states_st, cur_c_states_st)
                       
        decoder_out_st, (new_h_states, new_c_states) = model.st_decoder(decoder_input_st, st_decode_state, need_pad=False)
        decoder_out_st = decoder_out_st.unsqueeze(1)
        decoder_out_st = model.st_joiner.decoder_proj(decoder_out_st)
        # decoder_out is of shape (num_hyps, 1, 1, joiner_dim)
        # st
        current_encoder_out_st = torch.index_select(
            current_encoder_out_st,
            dim=0,
            index=hyps_shape_st.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits_st = model.st_joiner(
            current_encoder_out_st,
            decoder_out_st,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits_st = logits_st.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        if use_hat == True:
            # For blank symbol, log-prob is log-sigmoid of the score
            logp_b_st = torch.nn.functional.logsigmoid(logits_st[..., 0])
            # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
            # need to add the following term to the log-probs of non-blank symbols. This
            # is equivalent to log(1 - sigmoid(logits[..., 0])).
            nb_shift_st = logp_b_st - logits_st[..., 0]
            nb_shift_st = nb_shift_st.unsqueeze(-1)
            log_probs1_st = (logits_st[..., 1:] / temperature).log_softmax(
                dim=-1
            ) + nb_shift_st  # (num_hyps, vocab_size-1)
            log_probs_st = torch.cat((logp_b_st.unsqueeze(-1), log_probs1_st), dim=-1)
            log_probs_st.add_(ys_log_probs_st)
        else:
            log_probs_st = (logits_st / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
            log_probs_st.add_(ys_log_probs_st)
            
        vocab_size_st = log_probs_st.size(-1)

        log_probs_st = log_probs_st.reshape(-1)
        # print(hyps_shape_st.row_splits(1) * vocab_size_st)
        # print(log_probs_st.numel())
        row_splits_st = hyps_shape_st.row_splits(1) * vocab_size_st
        log_probs_shape_st = k2.ragged.create_ragged_shape2(
            row_splits=row_splits_st, cached_tot_size=log_probs_st.numel()
        )
        ragged_log_probs_st = k2.RaggedTensor(shape=log_probs_shape_st, value=log_probs_st)

        for i in range(batch_size):
            topk_log_probs_st, topk_indexes_st = ragged_log_probs_st[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                topk_hyp_indexes_st = (topk_indexes_st // vocab_size_st).tolist()
                topk_token_indexes_st = (topk_indexes_st % vocab_size_st).tolist()

            for k in range(len(topk_hyp_indexes_st)):
                hyp_idx = topk_hyp_indexes_st[k]
                hyp_st = A_st[i][hyp_idx]

                new_ys_st = hyp_st.ys[:]
                new_token = topk_token_indexes_st[k]
                new_timestamp_st = hyp_st.timestamp[:]
                if new_token not in (blank_id_st, unk_id_st):
                    new_ys_st.append(new_token)
                    new_timestamp_st.append(t)

                new_log_prob_st = topk_log_probs_st[k]
                
            #      Hypothesis(
            #     ys=[blank_id_st],
            #     log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            #     timestamp=[],
            #     decoder_state=[(h, c)],
            # )
            # LSTM decoder_state 
                if new_token == blank_id_st:
                    new_hyp_st = Hypothesis(
                        ys=new_ys_st,
                        log_prob=new_log_prob_st,
                        timestamp=new_timestamp_st,
                        decoder_state=hyp_st.decoder_state,  # Keep old state
                    )
                else:
                    # Take the updated state for this hypothesis
                    new_hyp_st = Hypothesis(
                        ys=new_ys_st,
                        log_prob=new_log_prob_st,
                        timestamp=new_timestamp_st,
                        decoder_state=[(
                            new_h_states[:, hyp_idx: hyp_idx + 1, :],
                            new_c_states[:, hyp_idx: hyp_idx + 1, :],
                        )],
                    )
                # new_hyp_st = Hypothesis(
                #     ys=new_ys_st, log_prob=new_log_prob_st, timestamp=new_timestamp_st
                # )
                # breakpoint()
                B_st[i].add(new_hyp_st)

    # ASR
    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    batch_size_list = packed_encoder_out.batch_sizes.tolist()
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
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]
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
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        # For blank symbol, log-prob is log-sigmoid of the score
        if use_hat == True:
          # For blank symbol, log-prob is log-sigmoid of the score
          logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
          # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
          # need to add the following term to the log-probs of non-blank symbols. This
          # is equivalent to log(1 - sigmoid(logits[..., 0])).
          #breakpoint()
          nb_shift = logp_b - logits[..., 0]
          nb_shift = nb_shift.unsqueeze(-1)
          log_probs1 = (logits[..., 1:] / temperature).log_softmax(dim=-1) + nb_shift # (num_hyps, vocab_size-1)
          log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1) 
          log_probs.add_(ys_log_probs)
        else:
          log_probs = (logits / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
          log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
    
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    
    B_st = B_st + finalized_B_st
    best_hyps_st = [b.get_most_probable(length_norm=True) for b in B_st]
    if st_lstm_pred:
        sorted_ans_st = [h.ys[1:] for h in best_hyps_st]
    else:
        sorted_ans_st = [h.ys for h in best_hyps_st]
    sorted_timestamps_st = [h.timestamp for h in best_hyps_st]
    ans_st = []
    ans_timestamps_st = []
    st_unsorted_indices = packed_st_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans_st.append(sorted_ans_st[st_unsorted_indices[i]])
        ans_timestamps_st.append(sorted_timestamps_st[st_unsorted_indices[i]])

    if not return_timestamps:
        return [ans, ans_st]
    else:
        return DecodingResults(
            hyps=[ans, ans_st],
            timestamps=[ans_timestamps, ans_timestamps_st],
        )


def modified_beam_search_lm_shallow_fusion(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    beam: int = 4,
    return_timestamps: bool = False,
    subtract_ilm: bool = True,
    ilm_scale: float = 0.1,
    temperature: float = 1.0,
) -> List[List[int]]:
    """Modified_beam_search + NN LM shallow fusion

    Args:
        model (Transducer):
            The transducer model
        encoder_out (torch.Tensor):
            Encoder output in (N,T,C)
        encoder_out_lens (torch.Tensor):
            A 1-D tensor of shape (N,), containing the number of
            valid frames in encoder_out before padding.
        sp:
            Sentence piece generator.
        LM (LmScorer):
            A neural net LM, e.g RNN or Transformer
        beam (int, optional):
            Beam size. Defaults to 4.

    Returns:
      Return a list-of-list of token IDs. ans[i] is the decoding results
      for the i-th utterance.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert LM is not None
    lm_scale = LM.lm_scale

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    sos_id = getattr(LM, "sos_id", 1)
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    # get initial lm score and lm state by scoring the "sos" token
    sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
    lens = torch.tensor([1]).to(device)
    init_score, init_states = LM.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state=init_states,
                lm_score=init_score.reshape(-1),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for (t, batch_size) in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]  # get batch
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        # For blank symbol, log-prob is log-sigmoid of the score
        logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
        # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
        # need to add the following term to the log-probs of non-blank symbols. This
        # is equivalent to log(1 - sigmoid(logits[..., 0])).
        nb_shift = logp_b - logits[..., 0]
        nb_shift = nb_shift.unsqueeze(-1)
        log_probs1 = (logits[..., 1:]).log_softmax(dim=-1) + nb_shift
        if subtract_ilm:
            ilm_logits = model.joiner(
                torch.zeros_like(
                    current_encoder_out, device=current_encoder_out.device
                ),
                decoder_out,
                project_input=False,
            )
            ilm_logits = ilm_logits.squeeze(1).squeeze(1)
            ilm_logp_b = torch.nn.functional.logsigmoid(ilm_logits[..., 0])
            ilm_nb_shift = ilm_logp_b - ilm_logits[..., 0]
            ilm_nb_shift = ilm_nb_shift.unsqueeze(-1)
            ilm_log_probs = (ilm_logits[..., 1:]).log_softmax(dim=-1) + ilm_nb_shift
            log_probs1 -= ilm_scale * ilm_log_probs

        log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1)
        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        """
        for all hyps with a non-blank new token, score this token.
        It is a little confusing here because this for-loop
        looks very similar to the one below. Here, we go through all
        top-k tokens and only add the non-blanks ones to the token_list.
        `LM` will score those tokens given the LM states. Note that
        the variable `scores` is the LM score after seeing the new
        non-blank token.
        """
        token_list = []  # a list of list
        hs = []
        cs = []
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    if LM.lm_type == "rnn":
                        token_list.append([new_token])
                        # store the LSTM states
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        # for transformer LM
                        token_list.append(
                            [sos_id] + hyp.ys[context_size:] + [new_token]
                        )

        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if LM.lm_type == "rnn":
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

            scores, lm_states = LM.score_token(tokens_to_score, x_lens, state)

        count = 0  # index, used to locate score and lm states
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]

                lm_score = hyp.lm_score
                state = hyp.state

                hyp_log_prob = topk_log_probs[k]  # get score of current hyp
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):

                    ys.append(new_token)
                    new_timestamp.append(t)

                    hyp_log_prob += lm_score[new_token] * lm_scale  # add the lm score

                    lm_score = scores[count]
                    if LM.lm_type == "rnn":
                        state = (
                            lm_states[0][:, count, :].unsqueeze(1),
                            lm_states[1][:, count, :].unsqueeze(1),
                        )
                    count += 1

                new_hyp = Hypothesis(
                    ys=ys,
                    log_prob=hyp_log_prob,
                    state=state,
                    lm_score=lm_score,
                    timestamp=new_timestamp,
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )
    

def modified_beam_search_lm_rescore_LODR(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    LODR_lm: NgramLm,
    sp: spm.SentencePieceProcessor,
    lm_scale_list: List[int],
    beam: int = 4,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcoded.
    Rescore the final results with RNNLM and return the one with the highest score

    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      LM:
        A neural network language model
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
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
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
        # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
        # need to add the following term to the log-probs of non-blank symbols. This
        # is equivalent to log(1 - sigmoid(logits[..., 0])).
        nb_shift = logp_b - logits[..., 0]
        nb_shift = nb_shift.unsqueeze(-1)
        log_probs1 = (logits[..., 1:]/ temperature).log_softmax(dim=-1) + nb_shift

        #log_probs = (logits / temperature).log_softmax(dim=-1)  # (num_hyps, vocab_size)
        log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1)

        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    # get the am_scores for n-best list
    hyps_shape = get_hyps_shape(B)
    am_scores = torch.tensor([hyp.log_prob.item() for b in B for hyp in b])
    am_scores = k2.RaggedTensor(value=am_scores, shape=hyps_shape).to(device)

    # now LM rescore
    # prepare input data to LM
    candidate_seqs = [hyp.ys[context_size:] for b in B for hyp in b]
    possible_seqs = k2.RaggedTensor(candidate_seqs)
    row_splits = possible_seqs.shape.row_splits(1)
    sentence_token_lengths = row_splits[1:] - row_splits[:-1]
    possible_seqs_with_sos = add_sos(possible_seqs, sos_id=1)
    possible_seqs_with_eos = add_eos(possible_seqs, eos_id=1)
    sentence_token_lengths += 1

    x = possible_seqs_with_sos.pad(mode="constant", padding_value=blank_id)
    y = possible_seqs_with_eos.pad(mode="constant", padding_value=blank_id)
    x = x.to(device).to(torch.int64)
    y = y.to(device).to(torch.int64)
    sentence_token_lengths = sentence_token_lengths.to(device).to(torch.int64)

    lm_scores = LM.lm(x=x, y=y, lengths=sentence_token_lengths)
    assert lm_scores.ndim == 2
    lm_scores = -1 * lm_scores.sum(dim=1)

    # now LODR scores
    import math

    LODR_scores = []
    for seq in candidate_seqs:
        tokens = " ".join(sp.id_to_piece(seq))
        LODR_scores.append(LODR_lm.score(tokens))
    LODR_scores = torch.tensor(LODR_scores).to(device) * math.log(
        10
    )  # arpa scores are 10-based
    assert lm_scores.shape == LODR_scores.shape

    ans = {}
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()

    LODR_scale_list = [0.02 * i for i in range(2, 10)]
    # get the best hyp with different lm_scale and lodr_scale
    for lm_scale in lm_scale_list:
        for lodr_scale in LODR_scale_list:
            key = f"nnlm_scale_{lm_scale:.2f}_lodr_scale_{lodr_scale:.2f}"
            tot_scores = (
                am_scores.values / lm_scale + lm_scores - LODR_scores * lodr_scale
            )
            ragged_tot_scores = k2.RaggedTensor(shape=am_scores.shape, value=tot_scores)
            max_indexes = ragged_tot_scores.argmax().tolist()
            unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
            hyps = []
            for idx in unsorted_indices:
                hyps.append(unsorted_hyps[idx])

            ans[key] = hyps
    return ans


def modified_beam_search_LODR(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LODR_lm: NgramLm,
    LODR_lm_scale: float,
    LM: LmScorer,
    beam: int = 4,
    context_graph: Optional[ContextGraph] = None,
) -> List[List[int]]:
    """This function implements LODR (https://arxiv.org/abs/2203.16776) with
    `modified_beam_search`. It uses a bi-gram language model as the estimate
    of the internal language model and subtracts its score during shallow fusion
    with an external language model. This implementation uses a RNNLM as the
    external language model.

    Args:
        model (Transducer):
            The transducer model
        encoder_out (torch.Tensor):
            Encoder output in (N,T,C)
        encoder_out_lens (torch.Tensor):
            A 1-D tensor of shape (N,), containing the number of
            valid frames in encoder_out before padding.
        LODR_lm:
            A low order n-gram LM, whose score will be subtracted during shallow fusion
        LODR_lm_scale:
            The scale of the LODR_lm
        LM:
            A neural net LM, e.g an RNNLM or transformer LM
        beam (int, optional):
            Beam size. Defaults to 4.

    Returns:
      Return a list-of-list of token IDs. ans[i] is the decoding results
      for the i-th utterance.

    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)
    assert LM is not None
    lm_scale = LM.lm_scale

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    sos_id = getattr(LM, "sos_id", 1)
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    # get initial lm score and lm state by scoring the "sos" token
    sos_token = torch.tensor([[sos_id]]).to(torch.int64).to(device)
    lens = torch.tensor([1]).to(device)
    init_score, init_states = LM.score_token(sos_token, lens)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state=init_states,  # state of the NN LM
                lm_score=init_score.reshape(-1),
                state_cost=NgramLmStateCost(
                    LODR_lm
                ),  # state of the source domain ngram
                context_state=None if context_graph is None else context_graph.root,
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]  # get batch
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # current_encoder_out's shape is (batch_size, 1, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)

        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )  # (num_hyps, 1, 1, vocab_size)

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)
        # For blank symbol, log-prob is log-sigmoid of the score
        logp_b = torch.nn.functional.logsigmoid(logits[..., 0])
        # Additionally, to ensure the the probs of blank and non-blank sum to 1, we
        # need to add the following term to the log-probs of non-blank symbols. This
        # is equivalent to log(1 - sigmoid(logits[..., 0])).
        nb_shift = logp_b - logits[..., 0]
        nb_shift = nb_shift.unsqueeze(-1)
        log_probs1 = (logits[..., 1:]).log_softmax(dim=-1) + nb_shift
        log_probs = torch.cat((logp_b.unsqueeze(-1), log_probs1), dim=-1)
        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)

        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)
        """
        for all hyps with a non-blank new token, score this token.
        It is a little confusing here because this for-loop
        looks very similar to the one below. Here, we go through all
        top-k tokens and only add the non-blanks ones to the token_list.
        LM will score those tokens given the LM states. Note that
        the variable `scores` is the LM score after seeing the new
        non-blank token.
        """
        token_list = []
        hs = []
        cs = []
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()
            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                new_token = topk_token_indexes[k]
                if new_token not in (blank_id, unk_id):
                    if LM.lm_type == "rnn":
                        token_list.append([new_token])
                        # store the LSTM states
                        hs.append(hyp.state[0])
                        cs.append(hyp.state[1])
                    else:
                        # for transformer LM
                        token_list.append(
                            [sos_id] + hyp.ys[context_size:] + [new_token]
                        )

        # forward NN LM to get new states and scores
        if len(token_list) != 0:
            x_lens = torch.tensor([len(tokens) for tokens in token_list]).to(device)
            if LM.lm_type == "rnn":
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

            scores, lm_states = LM.score_token(tokens_to_score, x_lens, state)

        count = 0  # index, used to locate score and lm states
        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]

                ys = hyp.ys[:]

                # current score of hyp
                lm_score = hyp.lm_score
                state = hyp.state

                hyp_log_prob = topk_log_probs[k]  # get score of current hyp
                new_token = topk_token_indexes[k]

                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state
                if new_token not in (blank_id, unk_id):
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                        ) = context_graph.forward_one_step(hyp.context_state, new_token)

                    ys.append(new_token)
                    state_cost = hyp.state_cost.forward_one_step(new_token)

                    # calculate the score of the latest token
                    current_ngram_score = state_cost.lm_score - hyp.state_cost.lm_score

                    assert current_ngram_score <= 0.0, (
                        state_cost.lm_score,
                        hyp.state_cost.lm_score,
                    )
                    # score = score + TDLM_score - LODR_score
                    # LODR_LM_scale should be a negative number here
                    hyp_log_prob += (
                        lm_score[new_token] * lm_scale
                        + LODR_lm_scale * current_ngram_score
                        + context_score
                    )  # add the lm score

                    lm_score = scores[count]
                    if LM.lm_type == "rnn":
                        state = (
                            lm_states[0][:, count, :].unsqueeze(1),
                            lm_states[1][:, count, :].unsqueeze(1),
                        )
                    count += 1
                else:
                    state_cost = hyp.state_cost

                new_hyp = Hypothesis(
                    ys=ys,
                    log_prob=hyp_log_prob,
                    state=state,
                    lm_score=lm_score,
                    state_cost=state_cost,
                    context_state=new_context_state,
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    # finalize context_state, if the matched contexts do not reach final state
    # we need to add the score on the corresponding backoff arc
    if context_graph is not None:
        finalized_B = [HypothesisList() for _ in range(len(B))]
        for i, hyps in enumerate(B):
            for hyp in list(hyps):
                context_score, new_context_state = context_graph.finalize(
                    hyp.context_state
                )
                finalized_B[i].add(
                    Hypothesis(
                        ys=hyp.ys,
                        log_prob=hyp.log_prob + context_score,
                        timestamp=hyp.timestamp,
                        context_state=new_context_state,
                    )
                )
        B = finalized_B

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans