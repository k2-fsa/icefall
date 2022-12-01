# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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
from dataclasses import dataclass
from typing import Dict, List, Optional

import k2
import torch
from model import Transducer

from icefall.decode import Nbest, one_best_decoding
from icefall.utils import get_texts


def fast_beam_search_one_best(
    model: Transducer,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
) -> List[List[int]]:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using modified beam search, and then
    the shortest path within the lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a HLG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the number of frames in `encoder_out`
        before padding.
      beam:
        Beam value, similar to the beam used in Kaldi..
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
    Returns:
      Return the decoded result.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
    )

    best_path = one_best_decoding(lattice)
    hyps = get_texts(best_path)
    return hyps


def fast_beam_search_nbest_oracle(
    model: Transducer,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    ref_texts: List[List[int]],
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
) -> List[List[int]]:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using modified beam search, and then
    we select `num_paths` linear paths from the lattice. The path
    that has the minimum edit distance with the given reference transcript
    is used as the output.

    This is the best result we can achieve for any nbest based rescoring
    methods.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a HLG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the number of frames in `encoder_out`
        before padding.
      beam:
        Beam value, similar to the beam used in Kaldi..
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
      num_paths:
        Number of paths to extract from the decoded lattice.
      ref_texts:
        A list-of-list of integers containing the reference transcripts.
        If the decoding_graph is a trivial_graph, the integer ID is the
        BPE token ID.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.

    Returns:
      Return the decoded result.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    hyps = nbest.build_levenshtein_graphs()
    refs = k2.levenshtein_graph(ref_texts, device=hyps.device)

    levenshtein_alignment = k2.levenshtein_alignment(
        refs=refs,
        hyps=hyps,
        hyp_to_ref_map=nbest.shape.row_ids(1),
        sorted_match_ref=True,
    )

    tot_scores = levenshtein_alignment.get_tot_scores(
        use_double_scores=False, log_semiring=False
    )
    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)

    max_indexes = ragged_tot_scores.argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)

    hyps = get_texts(best_path)
    return hyps


def fast_beam_search(
    model: Transducer,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
) -> k2.Fsa:
    """It limits the maximum number of symbols per frame to 1.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a HLG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the number of frames in `encoder_out`
        before padding.
      beam:
        Beam value, similar to the beam used in Kaldi..
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
    Returns:
      Return an FsaVec with axes [utt][state][arc] containing the decoded
      lattice. Note: When the input graph is a TrivialGraph, the returned
      lattice is actually an acceptor.
    """
    assert encoder_out.ndim == 3

    context_size = model.decoder.context_size
    vocab_size = model.decoder.vocab_size

    B, T, C = encoder_out.shape

    config = k2.RnntDecodingConfig(
        vocab_size=vocab_size,
        decoder_history_len=context_size,
        beam=beam,
        max_contexts=max_contexts,
        max_states=max_states,
    )
    individual_streams = []
    for i in range(B):
        individual_streams.append(k2.RnntDecodingStream(decoding_graph))
    decoding_streams = k2.RnntDecodingStreams(individual_streams, config)

    encoder_out_len = torch.ones(1, dtype=torch.int32)
    decoder_out_len = torch.ones(1, dtype=torch.int32)

    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder(contexts, need_pad=False)
        # current_encoder_out is of shape
        # (shape.NumElements(), 1, joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            encoder_out_len.expand(decoder_out.size(0)),
            decoder_out_len.expand(decoder_out.size(0)),
        )  # (N, vocab_size)
        log_probs = logits.log_softmax(dim=-1)
        decoding_streams.advance(log_probs)
    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(encoder_out_lens.tolist())

    return lattice


def greedy_search(
    model: Transducer, encoder_out: torch.Tensor, max_sym_per_frame: int
) -> List[int]:
    """Greedy search for a single utterance.
    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      max_sym_per_frame:
        Maximum number of symbols per frame. If it is set to 0, the WER
        would be 100%.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    decoder_input = torch.tensor(
        [blank_id] * context_size, device=device, dtype=torch.int64
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)

    T = encoder_out.size(1)
    t = 0
    hyp = [blank_id] * context_size

    # Maximum symbols per utterance.
    max_sym_per_utt = 1000

    # symbols per frame
    sym_per_frame = 0

    # symbols per utterance decoded so far
    sym_per_utt = 0

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    while t < T and sym_per_utt < max_sym_per_utt:
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue

        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        logits = model.joiner(
            current_encoder_out, decoder_out, encoder_out_len, decoder_out_len
        )
        # logits is (1, vocab_size)

        y = logits.argmax().item()
        if y != blank_id:
            hyp.append(y)
            decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                1, context_size
            )

            decoder_out = model.decoder(decoder_input, need_pad=False)

            sym_per_utt += 1
            sym_per_frame += 1
        else:
            sym_per_frame = 0
            t += 1
    hyp = hyp[context_size:]  # remove blanks

    return hyp


def greedy_search_batch(
    model: Transducer,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
) -> List[List[int]]:
    """Greedy search in batch mode. It hardcodes --max-sym-per-frame=1.
    Args:
      model:
        The transducer model.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
    Returns:
      Return a list-of-list of token IDs containing the decoded results.
      len(ans) equals to encoder_out.size(0).
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
    context_size = model.decoder.context_size

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    hyps = [[blank_id] * context_size for _ in range(N)]

    decoder_input = torch.tensor(
        hyps,
        device=device,
        dtype=torch.int64,
    )  # (N, context_size)
    decoder_out = model.decoder(decoder_input, need_pad=False)
    # decoder_out: (N, 1, decoder_out_dim)

    encoder_out_len = torch.ones(1, dtype=torch.int32)
    decoder_out_len = torch.ones(1, dtype=torch.int32)

    encoder_out = packed_encoder_out.data

    offset = 0
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1)
        # current_encoder_out's shape: (batch_size, 1, encoder_out_dim)
        offset = end

        decoder_out = decoder_out[:batch_size]

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            encoder_out_len.expand(batch_size),
            decoder_out_len.expand(batch_size),
        )  # (batch_size, vocab_size)

        assert logits.ndim == 2, logits.shape
        y = logits.argmax(dim=1).tolist()
        emitted = False
        for i, v in enumerate(y):
            if v != blank_id:
                hyps[i].append(v)
                emitted = True

        if emitted:
            # update decoder output
            decoder_input = [h[-context_size:] for h in hyps[:batch_size]]
            decoder_input = torch.tensor(
                decoder_input,
                device=device,
                dtype=torch.int64,
            )  # (batch_size, context_size)
            decoder_out = model.decoder(
                decoder_input,
                need_pad=False,
            )  # (batch_size, 1, decoder_out_dim)

    sorted_ans = [h[context_size:] for h in hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans


@dataclass
class Hypothesis:
    # The predicted tokens so far.
    # Newly predicted tokens are appended to `ys`.
    ys: List[int]

    # The log prob of ys.
    # It contains only one entry.
    log_prob: torch.Tensor

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

    def topk(self, k: int) -> "HypothesisList":
        """Return the top-k hypothesis."""
        hyps = list(self._data.items())

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


def run_decoder(
    ys: List[int],
    model: Transducer,
    decoder_cache: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run the neural decoder model for a given hypothesis.

    Args:
      ys:
        The current hypothesis.
      model:
        The transducer model.
      decoder_cache:
        Cache to save computations.
    Returns:
      Return a 1-D tensor of shape (decoder_out_dim,) containing
      output of `model.decoder`.
    """
    context_size = model.decoder.context_size
    key = "_".join(map(str, ys[-context_size:]))
    if key in decoder_cache:
        return decoder_cache[key]

    device = model.device

    decoder_input = torch.tensor(
        [ys[-context_size:]],
        device=device,
        dtype=torch.int64,
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_cache[key] = decoder_out

    return decoder_out


def run_joiner(
    key: str,
    model: Transducer,
    encoder_out: torch.Tensor,
    decoder_out: torch.Tensor,
    encoder_out_len: torch.Tensor,
    decoder_out_len: torch.Tensor,
    joint_cache: Dict[str, torch.Tensor],
):
    """Run the joint network given outputs from the encoder and decoder.

    Args:
      key:
        A key into the `joint_cache`.
      model:
        The transducer model.
      encoder_out:
        A tensor of shape (1, 1, encoder_out_dim).
      decoder_out:
        A tensor of shape (1, 1, decoder_out_dim).
      encoder_out_len:
        A tensor with value [1].
      decoder_out_len:
        A tensor with value [1].
      joint_cache:
        A dict to save computations.
    Returns:
      Return a tensor from the output of log-softmax.
      Its shape is (vocab_size,).
    """
    if key in joint_cache:
        return joint_cache[key]

    logits = model.joiner(
        encoder_out,
        decoder_out,
        encoder_out_len,
        decoder_out_len,
    )

    # TODO(fangjun): Scale the blank posterior
    log_prob = logits.log_softmax(dim=-1)
    # log_prob is (1, 1, 1, vocab_size)

    log_prob = log_prob.squeeze()
    # Now log_prob is (vocab_size,)

    joint_cache[key] = log_prob

    return log_prob


def _get_hyps_shape(hyps: List[HypothesisList]) -> k2.RaggedShape:
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
    model: Transducer,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
) -> List[List[int]]:
    """Beam search in batch mode with --max-sym-per-frame=1 being hardcodded.

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
    Returns:
      Return a list-of-list of token IDs. ans[i] is the decoding results
      for the i-th utterance.
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
                ys=[blank_id] * context_size,
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            )
        )

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    encoder_out = packed_encoder_out.data
    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out.data[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1)
        # current_encoder_out's shape is: (batch_size, 1, encoder_out_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = _get_hyps_shape(B).to(device)

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

        decoder_out = model.decoder(decoder_input, need_pad=False)
        # decoder_output is of shape (num_hyps, 1, decoder_output_dim)

        # Note: For torch 1.7.1 and below, it requires a torch.int64 tensor
        # as index, so we use `to(torch.int64)` below.
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            encoder_out_len.expand(decoder_out.size(0)),
            decoder_out_len.expand(decoder_out.size(0)),
        )
        # logits is of shape (num_hyps, vocab_size)

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
                if new_token != blank_id:
                    new_ys.append(new_token)

                new_log_prob = topk_log_probs[k]
                new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

    return ans


def _deprecated_modified_beam_search(
    model: Transducer,
    encoder_out: torch.Tensor,
    beam: int = 4,
) -> List[int]:
    """It limits the maximum number of symbols per frame to 1.

    It decodes only one utterance at a time. We keep it only for reference.
    The function :func:`modified_beam_search` should be preferred as it
    supports batch decoding.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      beam:
        Beam size.
    Returns:
      Return the decoded result.
    """

    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    T = encoder_out.size(1)

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[blank_id] * context_size,
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
        )
    )

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    for t in range(T):
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # current_encoder_out is of shape (1, 1, encoder_out_dim)
        # fmt: on
        A = list(B)
        B = HypothesisList()

        ys_log_probs = torch.cat([hyp.log_prob.reshape(1, 1) for hyp in A])
        # ys_log_probs is of shape (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyp in A],
            device=device,
        )
        # decoder_input is of shape (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False)
        # decoder_output is of shape (num_hyps, 1, decoder_output_dim)

        current_encoder_out = current_encoder_out.expand(decoder_out.size(0), 1, -1)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            encoder_out_len.expand(decoder_out.size(0)),
            decoder_out_len.expand(decoder_out.size(0)),
        )
        # logits is of shape (num_hyps, vocab_size)
        log_probs = logits.log_softmax(dim=-1)

        log_probs.add_(ys_log_probs)

        log_probs = log_probs.reshape(-1)
        topk_log_probs, topk_indexes = log_probs.topk(beam)

        # topk_hyp_indexes are indexes into `A`
        topk_hyp_indexes = topk_indexes // logits.size(-1)
        topk_token_indexes = topk_indexes % logits.size(-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            topk_hyp_indexes = topk_hyp_indexes.tolist()
            topk_token_indexes = topk_token_indexes.tolist()

        for i in range(len(topk_hyp_indexes)):
            hyp = A[topk_hyp_indexes[i]]
            new_ys = hyp.ys[:]
            new_token = topk_token_indexes[i]
            if new_token != blank_id:
                new_ys.append(new_token)
            new_log_prob = topk_log_probs[i]
            new_hyp = Hypothesis(ys=new_ys, log_prob=new_log_prob)
            B.add(new_hyp)

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  # [context_size:] to remove blanks

    return ys


def beam_search(
    model: Transducer,
    encoder_out: torch.Tensor,
    beam: int = 4,
) -> List[int]:
    """
    It implements Algorithm 1 in https://arxiv.org/pdf/1211.3711.pdf

    espnet/nets/beam_search_transducer.py#L247 is used as a reference.

    Args:
      model:
        An instance of `Transducer`.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder. Support only N==1 for now.
      beam:
        Beam size.
    Returns:
      Return the decoded result.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    device = model.device

    decoder_input = torch.tensor([blank_id] * context_size, device=device).reshape(
        1, context_size
    )

    decoder_out = model.decoder(decoder_input, need_pad=False)

    T = encoder_out.size(1)
    t = 0

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[blank_id] * context_size,
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
        )
    )

    max_sym_per_utt = 20000

    sym_per_utt = 0

    encoder_out_len = torch.tensor([1])
    decoder_out_len = torch.tensor([1])

    decoder_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :]
        # fmt: on
        A = B
        B = HypothesisList()

        joint_cache: Dict[str, torch.Tensor] = {}

        while True:
            y_star = A.get_most_probable()
            A.remove(y_star)

            decoder_out = run_decoder(
                ys=y_star.ys, model=model, decoder_cache=decoder_cache
            )

            key = "_".join(map(str, y_star.ys[-context_size:]))
            key += f"-t-{t}"
            log_prob = run_joiner(
                key=key,
                model=model,
                encoder_out=current_encoder_out,
                decoder_out=decoder_out,
                encoder_out_len=encoder_out_len,
                decoder_out_len=decoder_out_len,
                joint_cache=joint_cache,
            )

            # First, process the blank symbol
            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob

            # ys[:] returns a copy of ys
            B.add(Hypothesis(ys=y_star.ys[:], log_prob=new_y_star_log_prob))

            # Second, process other non-blank labels
            values, indices = log_prob.topk(beam + 1)
            for idx in range(values.size(0)):
                i = indices[idx].item()
                if i == blank_id:
                    continue

                new_ys = y_star.ys + [i]

                new_log_prob = y_star.log_prob + values[idx]
                A.add(Hypothesis(ys=new_ys, log_prob=new_log_prob))

            # Check whether B contains more than "beam" elements more probable
            # than the most probable in A
            A_most_probable = A.get_most_probable()

            kept_B = B.filter(A_most_probable.log_prob)

            if len(kept_B) >= beam:
                B = kept_B.topk(beam)
                break

        t += 1

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  # [context_size:] to remove blanks
    return ys
