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


def fast_beam_search_one_best(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    temperature: float = 1.0,
    ilme_scale: float = 0.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using fast beam search, and then
    the shortest path within the lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
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
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        ilme_scale=ilme_scale,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
    )

    best_path = one_best_decoding(lattice)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)


def fast_beam_search_nbest_LG(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    nbest_scale: float = 0.5,
    use_double_scores: bool = True,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    ilme_scale: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """It limits the maximum number of symbols per frame to 1.

    The process to get the results is:
     - (1) Use fast beam search to get a lattice
     - (2) Select `num_paths` paths from the lattice using k2.random_paths()
     - (3) Unique the selected paths
     - (4) Intersect the selected paths with the lattice and compute the
           shortest path from the intersection result
     - (5) The path with the largest score is used as the decoding output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
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
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
        ilme_scale=ilme_scale,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    # The following code is modified from nbest.intersect()
    word_fsa = k2.invert(nbest.fsa)
    if hasattr(lattice, "aux_labels"):
        # delete token IDs as it is not needed
        del word_fsa.aux_labels
    word_fsa.scores.zero_()
    word_fsa_with_epsilon_loops = k2.linear_fsa_with_self_loops(word_fsa)
    path_to_utt_map = nbest.shape.row_ids(1)

    if hasattr(lattice, "aux_labels"):
        # lattice has token IDs as labels and word IDs as aux_labels.
        # inv_lattice has word IDs as labels and token IDs as aux_labels
        inv_lattice = k2.invert(lattice)
        inv_lattice = k2.arc_sort(inv_lattice)
    else:
        inv_lattice = k2.arc_sort(lattice)

    if inv_lattice.shape[0] == 1:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=torch.zeros_like(path_to_utt_map),
            sorted_match_a=True,
        )
    else:
        path_lattice = k2.intersect_device(
            inv_lattice,
            word_fsa_with_epsilon_loops,
            b_to_a_map=path_to_utt_map,
            sorted_match_a=True,
        )

    # path_lattice has word IDs as labels and token IDs as aux_labels
    path_lattice = k2.top_sort(k2.connect(path_lattice))
    tot_scores = path_lattice.get_tot_scores(
        use_double_scores=use_double_scores,
        log_semiring=True,  # Note: we always use True
    )
    # See https://github.com/k2-fsa/icefall/pull/420 for why
    # we always use log_semiring=True

    ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
    best_hyp_indexes = ragged_tot_scores.argmax()
    best_path = k2.index_fsa(nbest.fsa, best_hyp_indexes)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)


def fast_beam_search_nbest(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    num_paths: int,
    nbest_scale: float = 0.5,
    use_double_scores: bool = True,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """It limits the maximum number of symbols per frame to 1.

    The process to get the results is:
     - (1) Use fast beam search to get a lattice
     - (2) Select `num_paths` paths from the lattice using k2.random_paths()
     - (3) Unique the selected paths
     - (4) Intersect the selected paths with the lattice and compute the
           shortest path from the intersection result
     - (5) The path with the largest score is used as the decoding output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
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
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        blank_penalty=blank_penalty,
        temperature=temperature,
        allow_partial=allow_partial,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )

    # at this point, nbest.fsa.scores are all zeros.

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa.scores contains acoustic scores

    max_indexes = nbest.tot_scores().argmax()

    best_path = k2.index_fsa(nbest.fsa, max_indexes)

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)


def fast_beam_search_nbest_oracle(
    model: nn.Module,
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
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
    allow_partial: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """It limits the maximum number of symbols per frame to 1.

    A lattice is first obtained using fast beam search, and then
    we select `num_paths` linear paths from the lattice. The path
    that has the minimum edit distance with the given reference transcript
    is used as the output.

    This is the best result we can achieve for any nbest based rescoring
    methods.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
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
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
        allow_partial=allow_partial,
        blank_penalty=blank_penalty,
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

    if not return_timestamps:
        return get_texts(best_path)
    else:
        return get_texts_with_timestamp(best_path)


def fast_beam_search(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    temperature: float = 1.0,
    subtract_ilme: bool = False,
    ilme_scale: float = 0.1,
    allow_partial: bool = False,
    blank_penalty: float = 0.0,
) -> k2.Fsa:
    """It limits the maximum number of symbols per frame to 1.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
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
      temperature:
        Softmax temperature.
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

    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        # shape is a RaggedShape of shape (B, context)
        # contexts is a Tensor of shape (shape.NumElements(), context_size)
        shape, contexts = decoding_streams.get_contexts()
        # `nn.Embedding()` in torch below v1.7.1 supports only torch.int64
        contexts = contexts.to(torch.int64)
        # decoder_out is of shape (shape.NumElements(), 1, decoder_out_dim)
        decoder_out = model.decoder(contexts, need_pad=False)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # current_encoder_out is of shape
        # (shape.NumElements(), 1, joiner_dim)
        # fmt: off
        current_encoder_out = torch.index_select(
            encoder_out[:, t:t + 1, :], 0, shape.row_ids(1).to(torch.int64)
        )
        # fmt: on
        logits = model.joiner(
            current_encoder_out.unsqueeze(2),
            decoder_out.unsqueeze(1),
            project_input=False,
        )
        logits = logits.squeeze(1).squeeze(1)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        log_probs = (logits / temperature).log_softmax(dim=-1)

        if ilme_scale != 0:
            ilme_logits = model.joiner(
                torch.zeros_like(
                    current_encoder_out, device=current_encoder_out.device
                ).unsqueeze(2),
                decoder_out.unsqueeze(1),
                project_input=False,
            )
            ilme_logits = ilme_logits.squeeze(1).squeeze(1)
            if blank_penalty != 0:
                ilme_logits[:, 0] -= blank_penalty
            ilme_log_probs = (ilme_logits / temperature).log_softmax(dim=-1)
            log_probs -= ilme_scale * ilme_log_probs

        decoding_streams.advance(log_probs)
    decoding_streams.terminate_and_flush_to_streams()
    lattice = decoding_streams.format_output(
        encoder_out_lens.tolist(), allow_partial=allow_partial
    )

    return lattice


def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    max_sym_per_frame: int,
    blank_penalty: float = 0.0,
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
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)

    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    unk_id = getattr(model, "unk_id", blank_id)

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], device=device, dtype=torch.int64
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

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
        if sym_per_frame >= max_sym_per_frame:
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

        if blank_penalty != 0:
            logits[:, :, :, 0] -= blank_penalty

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
    hyp = hyp[context_size:]  # remove blanks

    if not return_timestamps:
        return hyp
    else:
        return DecodingResults(
            hyps=[hyp],
            timestamps=[timestamp],
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
    context_graph: Optional[ContextGraph] = None,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
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
                context_state=None if context_graph is None else context_graph.root,
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

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

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
                context_score = 0
                new_context_state = None if context_graph is None else hyp.context_state
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)
                    if context_graph is not None:
                        (
                            context_score,
                            new_context_state,
                        ) = context_graph.forward_one_step(hyp.context_state, new_token)

                new_log_prob = topk_log_probs[k] + context_score

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
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


def modified_beam_search_lm_rescore(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
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

    ans = {}
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()

    # get the best hyp with different lm_scale
    for lm_scale in lm_scale_list:
        key = f"nnlm_scale_{lm_scale:.2f}"
        tot_scores = am_scores.values + lm_scores * lm_scale
        ragged_tot_scores = k2.RaggedTensor(shape=am_scores.shape, value=tot_scores)
        max_indexes = ragged_tot_scores.argmax().tolist()
        unsorted_hyps = [candidate_seqs[idx] for idx in max_indexes]
        hyps = []
        for idx in unsorted_indices:
            hyps.append(unsorted_hyps[idx])

        ans[key] = hyps
    return ans


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

    LODR_scale_list = [0.05 * i for i in range(1, 20)]
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


def _deprecated_modified_beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
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
      return_timestamps:
        Whether to return timestamps.

    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """

    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    device = next(model.parameters()).device

    T = encoder_out.size(1)

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[-1] * (context_size - 1) + [blank_id],
            log_prob=torch.zeros(1, dtype=torch.float32, device=device),
            timestamp=[],
        )
    )
    encoder_out = model.joiner.encoder_proj(encoder_out)

    for t in range(T):
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        # current_encoder_out is of shape (1, 1, 1, encoder_out_dim)
        # fmt: on
        A = list(B)
        B = HypothesisList()

        ys_log_probs = torch.cat([hyp.log_prob.reshape(1, 1) for hyp in A])
        # ys_log_probs is of shape (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyp in A],
            device=device,
            dtype=torch.int64,
        )
        # decoder_input is of shape (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out = model.joiner.decoder_proj(decoder_out)
        # decoder_output is of shape (num_hyps, 1, 1, joiner_dim)

        current_encoder_out = current_encoder_out.expand(
            decoder_out.size(0), 1, 1, -1
        )  # (num_hyps, 1, 1, encoder_out_dim)

        logits = model.joiner(
            current_encoder_out,
            decoder_out,
            project_input=False,
        )
        # logits is of shape (num_hyps, 1, 1, vocab_size)
        logits = logits.squeeze(1).squeeze(1)

        # now logits is of shape (num_hyps, vocab_size)
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
            new_timestamp = hyp.timestamp[:]
            new_token = topk_token_indexes[i]
            if new_token not in (blank_id, unk_id):
                new_ys.append(new_token)
                new_timestamp.append(t)
            new_log_prob = topk_log_probs[i]
            new_hyp = Hypothesis(
                ys=new_ys, log_prob=new_log_prob, timestamp=new_timestamp
            )
            B.add(new_hyp)

    best_hyp = B.get_most_probable(length_norm=True)
    ys = best_hyp.ys[context_size:]  # [context_size:] to remove blanks

    if not return_timestamps:
        return ys
    else:
        return DecodingResults(hyps=[ys], timestamps=[best_hyp.timestamp])


def beam_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
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
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.

    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    assert encoder_out.ndim == 3

    # support only batch_size == 1 for now
    assert encoder_out.size(0) == 1, encoder_out.size(0)
    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size

    device = next(model.parameters()).device

    decoder_input = torch.tensor(
        [blank_id] * context_size,
        device=device,
        dtype=torch.int64,
    ).reshape(1, context_size)

    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)

    encoder_out = model.joiner.encoder_proj(encoder_out)

    T = encoder_out.size(1)
    t = 0

    B = HypothesisList()
    B.add(
        Hypothesis(
            ys=[-1] * (context_size - 1) + [blank_id], log_prob=0.0, timestamp=[]
        )
    )

    max_sym_per_utt = 20000

    sym_per_utt = 0

    decoder_cache: Dict[str, torch.Tensor] = {}

    while t < T and sym_per_utt < max_sym_per_utt:
        # fmt: off
        current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
        # fmt: on
        A = B
        B = HypothesisList()

        joint_cache: Dict[str, torch.Tensor] = {}

        # TODO(fangjun): Implement prefix search to update the `log_prob`
        # of hypotheses in A

        while True:
            y_star = A.get_most_probable()
            A.remove(y_star)

            cached_key = y_star.key

            if cached_key not in decoder_cache:
                decoder_input = torch.tensor(
                    [y_star.ys[-context_size:]],
                    device=device,
                    dtype=torch.int64,
                ).reshape(1, context_size)

                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
                decoder_cache[cached_key] = decoder_out
            else:
                decoder_out = decoder_cache[cached_key]

            cached_key += f"-t-{t}"
            if cached_key not in joint_cache:
                logits = model.joiner(
                    current_encoder_out,
                    decoder_out.unsqueeze(1),
                    project_input=False,
                )

                if blank_penalty != 0:
                    logits[:, :, :, 0] -= blank_penalty

                # TODO(fangjun): Scale the blank posterior
                log_prob = (logits / temperature).log_softmax(dim=-1)
                # log_prob is (1, 1, 1, vocab_size)
                log_prob = log_prob.squeeze()
                # Now log_prob is (vocab_size,)
                joint_cache[cached_key] = log_prob
            else:
                log_prob = joint_cache[cached_key]

            # First, process the blank symbol
            skip_log_prob = log_prob[blank_id]
            new_y_star_log_prob = y_star.log_prob + skip_log_prob

            # ys[:] returns a copy of ys
            B.add(
                Hypothesis(
                    ys=y_star.ys[:],
                    log_prob=new_y_star_log_prob,
                    timestamp=y_star.timestamp[:],
                )
            )

            # Second, process other non-blank labels
            values, indices = log_prob.topk(beam + 1)
            for i, v in zip(indices.tolist(), values.tolist()):
                if i in (blank_id, unk_id):
                    continue
                new_ys = y_star.ys + [i]
                new_log_prob = y_star.log_prob + v
                new_timestamp = y_star.timestamp + [t]
                A.add(
                    Hypothesis(
                        ys=new_ys,
                        log_prob=new_log_prob,
                        timestamp=new_timestamp,
                    )
                )

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

    if not return_timestamps:
        return ys
    else:
        return DecodingResults(hyps=[ys], timestamps=[best_hyp.timestamp])


def fast_beam_search_with_nbest_rescoring(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    ngram_lm_scale_list: List[float],
    num_paths: int,
    G: k2.Fsa,
    sp: spm.SentencePieceProcessor,
    word_table: k2.SymbolTable,
    oov_word: str = "<UNK>",
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Dict[str, Union[List[List[int]], DecodingResults]]:
    """It limits the maximum number of symbols per frame to 1.
    A lattice is first obtained using fast beam search, num_path are selected
    and rescored using a given language model. The shortest path within the
    lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the number of frames in `encoder_out`
        before padding.
      beam:
        Beam value, similar to the beam used in Kaldi.
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
      ngram_lm_scale_list:
        A list of floats representing LM score scales.
      num_paths:
        Number of paths to extract from the decoded lattice.
      G:
        An FsaVec containing only a single FSA. It is an n-gram LM.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      oov_word:
        OOV words are replaced with this word.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      Return the decoded result in a dict, where the key has the form
      'ngram_lm_scale_xx' and the value is the decoded results
      optionally with timestamps. `xx` is the ngram LM scale value
      used during decoding, i.e., 0.1.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # at this point, nbest.fsa.scores are all zeros.

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa.scores contains acoustic scores

    am_scores = nbest.tot_scores()

    # Now we need to compute the LM scores of each path.
    # (1) Get the token IDs of each Path. We assume the decoding_graph
    # is an acceptor, i.e., lattice is also an acceptor
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)  # [path][arc]

    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.labels.contiguous())
    tokens = tokens.remove_values_leq(0)  # remove -1 and 0

    token_list: List[List[int]] = tokens.tolist()
    word_list: List[List[str]] = sp.decode(token_list)

    assert isinstance(oov_word, str), oov_word
    assert oov_word in word_table, oov_word
    oov_word_id = word_table[oov_word]

    word_ids_list: List[List[int]] = []

    for words in word_list:
        this_word_ids = []
        for w in words.split():
            if w in word_table:
                this_word_ids.append(word_table[w])
            else:
                this_word_ids.append(oov_word_id)
        word_ids_list.append(this_word_ids)

    word_fsas = k2.linear_fsa(word_ids_list, device=lattice.device)
    word_fsas_with_self_loops = k2.add_epsilon_self_loops(word_fsas)

    num_unique_paths = len(word_ids_list)

    b_to_a_map = torch.zeros(
        num_unique_paths,
        dtype=torch.int32,
        device=lattice.device,
    )

    rescored_word_fsas = k2.intersect_device(
        a_fsas=G,
        b_fsas=word_fsas_with_self_loops,
        b_to_a_map=b_to_a_map,
        sorted_match_a=True,
        ret_arc_maps=False,
    )

    rescored_word_fsas = k2.remove_epsilon_self_loops(rescored_word_fsas)
    rescored_word_fsas = k2.top_sort(k2.connect(rescored_word_fsas))
    ngram_lm_scores = rescored_word_fsas.get_tot_scores(
        use_double_scores=True,
        log_semiring=False,
    )

    ans: Dict[str, Union[List[List[int]], DecodingResults]] = {}
    for s in ngram_lm_scale_list:
        key = f"ngram_lm_scale_{s}"
        tot_scores = am_scores.values + s * ngram_lm_scores
        ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)

        if not return_timestamps:
            ans[key] = get_texts(best_path)
        else:
            ans[key] = get_texts_with_timestamp(best_path)

    return ans


def fast_beam_search_with_nbest_rnn_rescoring(
    model: nn.Module,
    decoding_graph: k2.Fsa,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: float,
    max_states: int,
    max_contexts: int,
    ngram_lm_scale_list: List[float],
    num_paths: int,
    G: k2.Fsa,
    sp: spm.SentencePieceProcessor,
    word_table: k2.SymbolTable,
    rnn_lm_model: torch.nn.Module,
    rnn_lm_scale_list: List[float],
    oov_word: str = "<UNK>",
    use_double_scores: bool = True,
    nbest_scale: float = 0.5,
    temperature: float = 1.0,
    return_timestamps: bool = False,
) -> Dict[str, Union[List[List[int]], DecodingResults]]:
    """It limits the maximum number of symbols per frame to 1.
    A lattice is first obtained using fast beam search, num_path are selected
    and rescored using a given language model and a rnn-lm.
    The shortest path within the lattice is used as the final output.

    Args:
      model:
        An instance of `Transducer`.
      decoding_graph:
        Decoding graph used for decoding, may be a TrivialGraph or a LG.
      encoder_out:
        A tensor of shape (N, T, C) from the encoder.
      encoder_out_lens:
        A tensor of shape (N,) containing the number of frames in `encoder_out`
        before padding.
      beam:
        Beam value, similar to the beam used in Kaldi.
      max_states:
        Max states per stream per frame.
      max_contexts:
        Max contexts pre stream per frame.
      ngram_lm_scale_list:
        A list of floats representing LM score scales.
      num_paths:
        Number of paths to extract from the decoded lattice.
      G:
        An FsaVec containing only a single FSA. It is an n-gram LM.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      rnn_lm_model:
        A rnn-lm model used for LM rescoring
      rnn_lm_scale_list:
        A list of floats representing RNN score scales.
      oov_word:
        OOV words are replaced with this word.
      use_double_scores:
        True to use double precision for computation. False to use
        single precision.
      nbest_scale:
        It's the scale applied to the lattice.scores. A smaller value
        yields more unique paths.
      temperature:
        Softmax temperature.
      return_timestamps:
        Whether to return timestamps.
    Returns:
      Return the decoded result in a dict, where the key has the form
      'ngram_lm_scale_xx' and the value is the decoded results
      optionally with timestamps. `xx` is the ngram LM scale value
      used during decoding, i.e., 0.1.
    """
    lattice = fast_beam_search(
        model=model,
        decoding_graph=decoding_graph,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=beam,
        max_states=max_states,
        max_contexts=max_contexts,
        temperature=temperature,
    )

    nbest = Nbest.from_lattice(
        lattice=lattice,
        num_paths=num_paths,
        use_double_scores=use_double_scores,
        nbest_scale=nbest_scale,
    )
    # at this point, nbest.fsa.scores are all zeros.

    nbest = nbest.intersect(lattice)
    # Now nbest.fsa.scores contains acoustic scores

    am_scores = nbest.tot_scores()

    # Now we need to compute the LM scores of each path.
    # (1) Get the token IDs of each Path. We assume the decoding_graph
    # is an acceptor, i.e., lattice is also an acceptor
    tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)  # [path][arc]

    tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.labels.contiguous())
    tokens = tokens.remove_values_leq(0)  # remove -1 and 0

    token_list: List[List[int]] = tokens.tolist()
    word_list: List[List[str]] = sp.decode(token_list)

    assert isinstance(oov_word, str), oov_word
    assert oov_word in word_table, oov_word
    oov_word_id = word_table[oov_word]

    word_ids_list: List[List[int]] = []

    for words in word_list:
        this_word_ids = []
        for w in words.split():
            if w in word_table:
                this_word_ids.append(word_table[w])
            else:
                this_word_ids.append(oov_word_id)
        word_ids_list.append(this_word_ids)

    word_fsas = k2.linear_fsa(word_ids_list, device=lattice.device)
    word_fsas_with_self_loops = k2.add_epsilon_self_loops(word_fsas)

    num_unique_paths = len(word_ids_list)

    b_to_a_map = torch.zeros(
        num_unique_paths,
        dtype=torch.int32,
        device=lattice.device,
    )

    rescored_word_fsas = k2.intersect_device(
        a_fsas=G,
        b_fsas=word_fsas_with_self_loops,
        b_to_a_map=b_to_a_map,
        sorted_match_a=True,
        ret_arc_maps=False,
    )

    rescored_word_fsas = k2.remove_epsilon_self_loops(rescored_word_fsas)
    rescored_word_fsas = k2.top_sort(k2.connect(rescored_word_fsas))
    ngram_lm_scores = rescored_word_fsas.get_tot_scores(
        use_double_scores=True,
        log_semiring=False,
    )

    # Now RNN-LM
    blank_id = model.decoder.blank_id
    sos_id = sp.piece_to_id("sos_id")
    eos_id = sp.piece_to_id("eos_id")

    sos_tokens = add_sos(tokens, sos_id)
    tokens_eos = add_eos(tokens, eos_id)
    sos_tokens_row_splits = sos_tokens.shape.row_splits(1)
    sentence_lengths = sos_tokens_row_splits[1:] - sos_tokens_row_splits[:-1]

    x_tokens = sos_tokens.pad(mode="constant", padding_value=blank_id)
    y_tokens = tokens_eos.pad(mode="constant", padding_value=blank_id)

    x_tokens = x_tokens.to(torch.int64)
    y_tokens = y_tokens.to(torch.int64)
    sentence_lengths = sentence_lengths.to(torch.int64)

    rnn_lm_nll = rnn_lm_model(x=x_tokens, y=y_tokens, lengths=sentence_lengths)
    assert rnn_lm_nll.ndim == 2
    assert rnn_lm_nll.shape[0] == len(token_list)
    rnn_lm_scores = -1 * rnn_lm_nll.sum(dim=1)

    ans: Dict[str, List[List[int]]] = {}
    for n_scale in ngram_lm_scale_list:
        for rnn_scale in rnn_lm_scale_list:
            key = f"ngram_lm_scale_{n_scale}_rnn_lm_scale_{rnn_scale}"
            tot_scores = (
                am_scores.values + n_scale * ngram_lm_scores + rnn_scale * rnn_lm_scores
            )
            ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
            max_indexes = ragged_tot_scores.argmax()
            best_path = k2.index_fsa(nbest.fsa, max_indexes)

            if not return_timestamps:
                ans[key] = get_texts(best_path)
            else:
                ans[key] = get_texts_with_timestamp(best_path)

    return ans


def modified_beam_search_ngram_rescoring(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ngram_lm: NgramLm,
    ngram_lm_scale: float,
    beam: int = 4,
    temperature: float = 1.0,
) -> List[List[int]]:
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
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device
    lm_scale = ngram_lm_scale

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
                state_cost=NgramLmStateCost(ngram_lm),
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for batch_size in batch_size_list:
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
            [
                hyp.log_prob.reshape(1, 1) + hyp.state_cost.lm_score * lm_scale
                for hyps in A
                for hyp in hyps
            ]
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
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    state_cost = hyp.state_cost.forward_one_step(new_token)
                else:
                    state_cost = hyp.state_cost

                # We only keep AM scores in new_hyp.log_prob
                new_log_prob = topk_log_probs[k] - hyp.state_cost.lm_score * lm_scale

                new_hyp = Hypothesis(
                    ys=new_ys, log_prob=new_log_prob, state_cost=state_cost
                )
                B[i].add(new_hyp)

    B = B + finalized_B
    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    ans = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])

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

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

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


def modified_beam_search_lm_shallow_fusion(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    LM: LmScorer,
    beam: int = 4,
    return_timestamps: bool = False,
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
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                state=init_states,
                lm_score=init_score.reshape(-1),
                timestamp=[],
            )
        )

    encoder_out = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    for t, batch_size in enumerate(batch_size_list):
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

        lm_scores = torch.cat(
            [hyp.lm_score.reshape(1, -1) for hyps in A for hyp in hyps]
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

        log_probs = logits.log_softmax(dim=-1)  # (num_hyps, vocab_size)

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
